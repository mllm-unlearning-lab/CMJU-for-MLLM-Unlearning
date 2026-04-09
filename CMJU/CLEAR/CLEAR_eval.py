import argparse
import json
import os
import random
import re
import traceback
from datetime import datetime

import torch
from datasets import load_dataset, load_from_disk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from peft import PeftModel
from rouge_score import rouge_scorer
from transformers import AutoProcessor, LlavaForConditionalGeneration

from data_process.CLEAR_process import (
    CLEARDataset,
    CAPTION_MODE,
    TEXT_QA_MODE,
)


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


random.seed(42)


def print_run_info(args: argparse.Namespace) -> None:
    print("=" * 100, flush=True)
    print(f"Eval start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("Arguments:", flush=True)
    for key, value in vars(args).items():
        print(f"  {key}: {value}", flush=True)
    print("=" * 100, flush=True)


def normalize_text(text: str) -> str:
    if text is None:
        return ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff ]", "", text)
    return text.strip()


def extract_choice_letter(text: str):
    """
    Extract the first valid option letter from model output.
    Supports:
    - A
    - B.
    - Option C
    - The answer is D
    """
    if not text:
        return None

    text = text.strip().upper()

    match = re.search(r"\b([A-Z])\b", text)
    if match:
        return match.group(1)

    if len(text) > 0 and text[0].isalpha():
        return text[0]

    return None


def compute_bleu(ground_truth: str, predicted_answer: str) -> float:
    reference = [ground_truth.split()]
    hypothesis = predicted_answer.split()
    smoothing_function = SmoothingFunction().method1
    return sentence_bleu(reference, hypothesis, smoothing_function=smoothing_function)


def formulate_prompt_with_options(question, options, answer):
    options_str = "\n".join([f"{chr(ord('A') + i)}. {value}" for i, value in enumerate(options)])
    gt = chr(ord('A') + options.index(answer))
    prompt = f"{question}\n{options_str}\nPlease answer with only the option letter."
    return prompt, gt


def load_eval_dataset(data_path: str, prefer_disk: bool = False):
    """
    Support two dataset formats:
    1. HuggingFace dataset script / local dataset repo: load_dataset(path, split="train")
    2. save_to_disk directory: load_from_disk(path)
    """
    if prefer_disk:
        return load_from_disk(data_path)

    try:
        return load_dataset(data_path, split="train")
    except Exception:
        return load_from_disk(data_path)


def load_model_and_processor(args):
    processor = AutoProcessor.from_pretrained(args.base_model_dir)

    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "right"
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

    if args.model_type == "lora":
        print(f"Loading base model from: {args.base_model_dir}", flush=True)
        base_model = LlavaForConditionalGeneration.from_pretrained(
            args.base_model_dir,
            torch_dtype=torch.float16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        print(f"Loading LoRA adapter from: {args.model_path}", flush=True)
        model = PeftModel.from_pretrained(
            base_model,
            args.model_path,
            is_trainable=False,
        )
    elif args.model_type == "full":
        print(f"Loading full model from: {args.model_path}", flush=True)
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
    else:
        raise ValueError(f"Unsupported model_type: {args.model_type}")

    model.eval()
    model.requires_grad_(False)
    return model, processor


def build_single_turn_prompt(processor, question: str, image=None):
    if image is None:
        conversation = [
            {
                "role": "user",
                "content": [{"type": "text", "text": question}],
            }
        ]
    else:
        conversation = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": question}],
            }
        ]
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    return prompt


def generate_answer(model, processor, question: str, image=None, max_new_tokens: int = 50):
    prompt = build_single_turn_prompt(processor, question, image=image)
    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt",
    )

    model_device = next(model.parameters()).device
    cast_dtype = torch.float16 if model_device.type == "cuda" else None

    processed_inputs = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            if cast_dtype is not None and torch.is_floating_point(v):
                processed_inputs[k] = v.to(model_device, dtype=cast_dtype)
            else:
                processed_inputs[k] = v.to(model_device)
        else:
            processed_inputs[k] = v

    with torch.no_grad():
        outputs = model.generate(
            **processed_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    out_wo_prompt = outputs[:, processed_inputs["input_ids"].shape[-1]:]
    generated_text = processor.tokenizer.decode(out_wo_prompt[0], skip_special_tokens=True).strip()
    return prompt, generated_text


def print_sample_log(task_name, idx, prompt, generated_text, ground_truth, is_correct, extra_info=None):
    print("\n" + "#" * 120, flush=True)
    print(f"[Task] {task_name}", flush=True)
    print(f"[Sample Index] {idx}", flush=True)
    if extra_info is not None:
        print(f"[Extra Info] {extra_info}", flush=True)
    print("-" * 120, flush=True)
    print("[Prompt]", flush=True)
    print(prompt, flush=True)
    print("-" * 120, flush=True)
    print("[Model Output]", flush=True)
    print(generated_text, flush=True)
    print("-" * 120, flush=True)
    print("[Ground Truth]", flush=True)
    print(ground_truth, flush=True)
    print("-" * 120, flush=True)
    print(f"[Correct] {is_correct}", flush=True)
    print("#" * 120 + "\n", flush=True)


def eval_classification(model, processor, data_path, with_options):
    print("################################## Classification Task Starts ##############################################", flush=True)
    print(f"Evaluating: {data_path}, with_options={with_options}", flush=True)

    if "forget" in data_path:
        vqa_data = load_eval_dataset(data_path, prefer_disk=False)
    elif "retain" in data_path:
        vqa_data = load_eval_dataset(data_path, prefer_disk=False)
    else:
        raise ValueError("Data path should contain 'forget' or 'retain'.")

    print(vqa_data, flush=True)

    correct_count = 0
    total_num = 0

    for idx, sample in enumerate(vqa_data):
        image = sample.get("image", None)
        question = sample.get("question", "What is the name of the person in the image?")
        answer = sample.get("name", "")

        if with_options:
            options = list(sample.get("perturbed_names", []))
            options.insert(random.randint(0, len(options)), answer)
            prompt_question, correct_answer = formulate_prompt_with_options(question, options, answer)
        else:
            prompt_question = question
            correct_answer = answer

        prompt, generated_text = generate_answer(
            model=model,
            processor=processor,
            question=prompt_question,
            image=image,
            max_new_tokens=50,
        )

        if with_options:
            predicted_answer = extract_choice_letter(generated_text)
            is_correct = predicted_answer == correct_answer
            extra_info = {"predicted_choice": predicted_answer, "correct_choice": correct_answer}
        else:
            normalized_pred = normalize_text(generated_text)
            normalized_gt = normalize_text(answer)
            is_correct = normalized_gt in normalized_pred if normalized_gt else False
            extra_info = {"normalized_pred": normalized_pred, "normalized_gt": normalized_gt}

        if is_correct:
            correct_count += 1

        total_num += 1

        print_sample_log(
            task_name="classification",
            idx=idx,
            prompt=prompt,
            generated_text=generated_text,
            ground_truth=answer,
            is_correct=is_correct,
            extra_info=extra_info,
        )

    accuracy = correct_count / total_num if total_num > 0 else 0.0
    print(f"Classification Correct Count: {correct_count}/{total_num}", flush=True)
    print(f"Classification Accuracy: {accuracy}", flush=True)
    print("################################## Classification Task Ends ##############################################", flush=True)

    return {
        "Accuracy": accuracy,
        "Correct Count": correct_count,
        "Total Count": total_num,
    }


def eval_classification_real(model, processor, data_path):
    print("################################## Real Classification Task Starts #########################################", flush=True)
    print(f"Evaluating: {data_path}", flush=True)

    df = load_eval_dataset(data_path, prefer_disk=False)

    correct_count = 0
    total_num = 0

    for idx, sample in enumerate(df):
        question = sample.get("question", "What is the name of the person in the image?")
        answer = sample.get("answer", "")
        options = list(sample.get("options", []))
        image = sample.get("image", None)

        options.insert(random.randint(0, len(options)), answer)
        prompt_question, correct_answer = formulate_prompt_with_options(question, options, answer)

        prompt, generated_text = generate_answer(
            model=model,
            processor=processor,
            question=prompt_question,
            image=image,
            max_new_tokens=50,
        )

        predicted_answer = extract_choice_letter(generated_text)
        is_correct = predicted_answer == correct_answer
        if is_correct:
            correct_count += 1
        total_num += 1

        extra_info = {"predicted_choice": predicted_answer, "correct_choice": correct_answer}

        print_sample_log(
            task_name="classification_real",
            idx=idx,
            prompt=prompt,
            generated_text=generated_text,
            ground_truth=answer,
            is_correct=is_correct,
            extra_info=extra_info,
        )

    accuracy = correct_count / total_num if total_num > 0 else 0.0
    print(f"Real Classification Correct Count: {correct_count}/{total_num}", flush=True)
    print(f"Real Classification Accuracy: {accuracy}", flush=True)
    print("################################## Real Classification Task Ends ###########################################", flush=True)

    return {
        "Accuracy": accuracy,
        "Correct Count": correct_count,
        "Total Count": total_num,
    }


def eval_generation(model, processor, data_path, mode):
    print("################################## Generation Task Starts ##################################################", flush=True)
    print(f"Evaluating generation on: {data_path}", flush=True)

    rouge_scorer_obj = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    df = load_eval_dataset(data_path, prefer_disk=False)

    vqa_data = CLEARDataset(df, mode=CAPTION_MODE)
    qa_data = CLEARDataset(df, mode=TEXT_QA_MODE)

    avg_scores = {}

    total_bleu_vqa = 0.0
    total_rouge1_vqa = 0.0
    total_rouge2_vqa = 0.0
    total_rougeL_vqa = 0.0
    total_vqa_num = 0

    for idx, sample in enumerate(vqa_data):
        image = sample["image"]
        question = sample["question"]
        answer = sample["answer"]

        prompt, generated_text = generate_answer(
            model=model,
            processor=processor,
            question=question,
            image=image,
            max_new_tokens=50,
        )

        bleu_score = compute_bleu(answer, generated_text)
        rouge_scores = rouge_scorer_obj.score(answer, generated_text)

        total_bleu_vqa += bleu_score
        total_rouge1_vqa += rouge_scores["rouge1"].fmeasure
        total_rouge2_vqa += rouge_scores["rouge2"].fmeasure
        total_rougeL_vqa += rouge_scores["rougeL"].fmeasure
        total_vqa_num += 1

        print_sample_log(
            task_name="generation_vqa",
            idx=idx,
            prompt=prompt,
            generated_text=generated_text,
            ground_truth=answer,
            is_correct="N/A",
            extra_info={
                "bleu": bleu_score,
                "rouge1": rouge_scores["rouge1"].fmeasure,
                "rouge2": rouge_scores["rouge2"].fmeasure,
                "rougeL": rouge_scores["rougeL"].fmeasure,
            },
        )

    print("[Generation] VQA loop finished.", flush=True)

    total_bleu_qa = 0.0
    total_rouge1_qa = 0.0
    total_rouge2_qa = 0.0
    total_rougeL_qa = 0.0
    total_qa_num = 0

    for idx, sample in enumerate(qa_data):
        image = sample["image"]
        question = sample["question"]
        answer = sample["answer"]

        prompt, generated_text = generate_answer(
            model=model,
            processor=processor,
            question=question,
            image=image,
            max_new_tokens=50,
        )

        bleu_score = compute_bleu(answer, generated_text)
        rouge_scores = rouge_scorer_obj.score(answer, generated_text)

        total_bleu_qa += bleu_score
        total_rouge1_qa += rouge_scores["rouge1"].fmeasure
        total_rouge2_qa += rouge_scores["rouge2"].fmeasure
        total_rougeL_qa += rouge_scores["rougeL"].fmeasure
        total_qa_num += 1

        print_sample_log(
            task_name="generation_qa",
            idx=idx,
            prompt=prompt,
            generated_text=generated_text,
            ground_truth=answer,
            is_correct="N/A",
            extra_info={
                "bleu": bleu_score,
                "rouge1": rouge_scores["rouge1"].fmeasure,
                "rouge2": rouge_scores["rouge2"].fmeasure,
                "rougeL": rouge_scores["rougeL"].fmeasure,
            },
        )

    print("[Generation] QA loop finished.", flush=True)

    avg_scores.update(
        {
            "Average ROUGE-1 (VQA)": total_rouge1_vqa / total_vqa_num if total_vqa_num > 0 else 0.0,
            "Average ROUGE-2 (VQA)": total_rouge2_vqa / total_vqa_num if total_vqa_num > 0 else 0.0,
            "Average ROUGE-L (VQA)": total_rougeL_vqa / total_vqa_num if total_vqa_num > 0 else 0.0,
            "Average BLEU (VQA)": total_bleu_vqa / total_vqa_num if total_vqa_num > 0 else 0.0,
            "Average ROUGE-1 (QA)": total_rouge1_qa / total_qa_num if total_qa_num > 0 else 0.0,
            "Average ROUGE-2 (QA)": total_rouge2_qa / total_qa_num if total_qa_num > 0 else 0.0,
            "Average ROUGE-L (QA)": total_rougeL_qa / total_qa_num if total_qa_num > 0 else 0.0,
            "Average BLEU (QA)": total_bleu_qa / total_qa_num if total_qa_num > 0 else 0.0,
        }
    )

    print("Generation metrics:", flush=True)
    print(avg_scores, flush=True)
    print("################################## Generation Task Ends ####################################################", flush=True)

    return avg_scores


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate LLaVA models on CLEAR datasets.")

    parser.add_argument("--base_model_dir", type=str, required=True, help="Base LLaVA model directory.")
    parser.add_argument("--model_path", type=str, required=True, help="Full model path or LoRA adapter path.")
    parser.add_argument("--model_type", type=str, required=True, choices=["full", "lora"])

    parser.add_argument("--data_folder", type=str, required=True, help="Root path of CLEAR data.")

    parser.add_argument("--forget_cls_folder", type=str, required=True, help="Forget classification dataset folder.")
    parser.add_argument("--forget_gen_folder", type=str, required=True, help="Forget generation dataset folder.")
    parser.add_argument("--retain_gen_folder", type=str, required=True, help="Retain generation dataset folder.")
    parser.add_argument("--retain_cls_folder", type=str, required=True, help="Retain classification dataset folder.")
    parser.add_argument("--realface_folder", type=str, required=True, help="Real face dataset folder.")
    parser.add_argument("--realworld_folder", type=str, required=True, help="Real world dataset folder.")

    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save evaluation outputs.")
    parser.add_argument("--output_file", type=str, required=True, help="Output file prefix.")
    parser.add_argument("--eval_list", type=str, required=True, help="String containing which splits to eval.")

    return parser.parse_args()


def main():
    try:
        args = parse_arguments()
        print_run_info(args)

        os.makedirs(args.output_folder, exist_ok=True)

        torch.cuda.empty_cache()
        model, processor = load_model_and_processor(args)
        torch.cuda.empty_cache()

        results_data = {}

        if "forget" in args.eval_list:
            print("### Evaluating Forget Set ###", flush=True)
            with_options = "perturbed" in args.forget_cls_folder.lower()

            forget_classification_result = eval_classification(
                model=model,
                processor=processor,
                data_path=f"{args.data_folder}/{args.forget_cls_folder}",
                with_options=with_options,
            )
            forget_generation_result = eval_generation(
                model=model,
                processor=processor,
                data_path=f"{args.data_folder}/{args.forget_gen_folder}",
                mode="forget",
            )

            results_data["Forget Set Results"] = {
                "classification": forget_classification_result,
                "generation": forget_generation_result,
            }

        if "retain" in args.eval_list:
            print("### Evaluating Retain Set ###", flush=True)
            with_options = "perturbed" in args.retain_cls_folder.lower()

            retain_classification_result = eval_classification(
                model=model,
                processor=processor,
                data_path=f"{args.data_folder}/{args.retain_cls_folder}",
                with_options=with_options,
            )
            retain_generation_result = eval_generation(
                model=model,
                processor=processor,
                data_path=f"{args.data_folder}/{args.retain_gen_folder}",
                mode="retain",
            )

            results_data["Retain Set Results"] = {
                "classification": retain_classification_result,
                "generation": retain_generation_result,
            }

        if "realface" in args.eval_list:
            print("### Evaluating Real Face Set ###", flush=True)
            realface_classification_result = eval_classification_real(
                model=model,
                processor=processor,
                data_path=f"{args.data_folder}/{args.realface_folder}",
            )
            results_data["Real Face Results"] = {
                "classification": realface_classification_result
            }

        if "realworld" in args.eval_list:
            print("### Evaluating Real World Set ###", flush=True)
            realworld_classification_result = eval_classification_real(
                model=model,
                processor=processor,
                data_path=f"{args.data_folder}/{args.realworld_folder}",
            )
            results_data["Real World Results"] = {
                "classification": realworld_classification_result
            }

        output_file = os.path.join(args.output_folder, f"{args.output_file}_final_evaluation_results.json")
        ensure_parent_dir(output_file)

        print(results_data, flush=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results_data, f, ensure_ascii=False, indent=4)

        print(f"Results saved to {output_file}", flush=True)

        config_file = os.path.join(args.output_folder, f"{args.output_file}_evalconfig.json")
        ensure_parent_dir(config_file)
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=4)

        print(f"Config saved to {config_file}", flush=True)

    except Exception as e:
        print(f"[FATAL ERROR] {repr(e)}", flush=True)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

