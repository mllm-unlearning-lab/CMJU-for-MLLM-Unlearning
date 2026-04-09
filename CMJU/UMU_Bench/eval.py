import os
import json
import sys
import ast
import difflib
import argparse
import re
from io import BytesIO
from datetime import datetime

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from peft import PeftModel
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from transformers import LlavaForConditionalGeneration, AutoProcessor, AutoTokenizer


# =============================================================================
# IO / Utility
# =============================================================================

def load_and_combine_parquet_files(directory):
    parquet_files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".parquet")
    ]
    combined_df = pd.concat(
        [pd.read_parquet(file) for file in parquet_files],
        ignore_index=True,
    )
    return combined_df


def compute_bleu(ground_truth, predicted_answer):
    reference = [ground_truth.split()]
    hypothesis = predicted_answer.split()
    smoothing_function = SmoothingFunction().method1
    return sentence_bleu(reference, hypothesis, smoothing_function=smoothing_function)


_TRAILING_PUNCT_TO_STRIP = set(list(".!?。！？,，;；:："))


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = s.replace("\u00A0", " ").replace("\u200B", " ").strip()
    s = re.sub(r"\s+", " ", s)
    while len(s) > 0 and s[-1] in _TRAILING_PUNCT_TO_STRIP:
        s = s[:-1].rstrip()
    return s


def strict_contains(ground_truth: str, response: str) -> bool:
    gt = normalize_text(ground_truth).lower()
    resp = normalize_text(response).lower()
    return gt in resp


def map_to_choice_text_v2(model_output: str, choices_text: list[str], options: dict):
    """
    Return:
        mapped_choice_text_or_None, match_method
    """
    norm_out = normalize_text(model_output).lower()
    norm_choices = [normalize_text(c).lower() for c in choices_text]

    # 1) exact contains
    for i, nc in enumerate(norm_choices):
        if nc and nc in norm_out:
            return choices_text[i], "exact_contains"

    # 2) letter match
    upper_out = model_output.strip().upper()
    m = re.match(r"^([ABCD])[.\s):：]?", upper_out)
    letter = m.group(1) if m else None
    if letter is None:
        m = re.search(r"\b([ABCD])\b", upper_out)
        if m:
            letter = m.group(1)

    if letter and letter in options:
        return options[letter], "letter_match"

    # 3) difflib threshold
    threshold = 0.5
    sims = [difflib.SequenceMatcher(None, norm_out, nc).ratio() for nc in norm_choices]
    best_i = sims.index(max(sims))
    if sims[best_i] >= threshold:
        return choices_text[best_i], f"difflib({sims[best_i]:.2f})"

    return None, "no_match"


def decode_new_tokens_llava(processor_or_tokenizer, output_ids, input_ids_len: int) -> str:
    gen_ids = output_ids[0][input_ids_len:]
    return processor_or_tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def get_model_device(model):
    return next(model.parameters()).device


# =============================================================================
# Model Loading
# =============================================================================

def load_model_processor_tokenizer(args):
    """
    model_type:
      - full: model_path is a full model directory
      - lora: model_path is a LoRA adapter directory, loaded on top of base_model_dir
    """
    print(f"[Init] Loading processor/tokenizer from: {args.base_model_dir}")
    processor = AutoProcessor.from_pretrained(args.base_model_dir, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_dir, local_files_only=True)

    if args.model_type == "full":
        print(f"[Init] Loading full model from: {args.model_path}")
        model = LlavaForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
    elif args.model_type == "lora":
        print(f"[Init] Loading base model from: {args.base_model_dir}")
        base_model = LlavaForConditionalGeneration.from_pretrained(
            args.base_model_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=True,
        )
        print(f"[Init] Loading LoRA adapter from: {args.model_path}")
        model = PeftModel.from_pretrained(base_model, args.model_path, local_files_only=True)
    else:
        raise ValueError(f"Unsupported model_type: {args.model_type}")

    model.eval()
    model.requires_grad_(False)
    return model, processor, tokenizer


# =============================================================================
# Evaluation: Classification
# =============================================================================

def evaluate_classification(
    parquet_file,
    processor,
    tokenizer,
    model,
    args,
    id_list_file=None,
    mode="default",
    forget_parquet_file=None,
):
    print("################################## Classification Task Starts ##############################################")
    print(f"############################## Evaluating {mode} Mode #########################################")

    if id_list_file:
        with open(id_list_file, "r") as f:
            id_list = json.load(f)
    elif mode == "test" and forget_parquet_file:
        forget_df = pd.read_parquet(forget_parquet_file)
        id_list = forget_df["ID"].unique().tolist()
    else:
        df = pd.read_parquet(parquet_file)
        id_list = df["ID"].unique().tolist()

    print(f"Loaded {len(id_list)} IDs from {id_list_file if id_list_file else 'parquet_file'}")

    total_image_textual_correct = 0
    total_image_textual_questions = 0
    total_pure_text_correct = 0
    total_pure_text_questions = 0
    unimodal = []
    multimodal = []

    if mode == "test":
        df = load_and_combine_parquet_files(parquet_file) if os.path.isdir(parquet_file) else pd.read_parquet(parquet_file)
    else:
        df = pd.read_parquet(parquet_file)

    eval_samples = df[df["ID"].isin(id_list)]
    device = get_model_device(model)

    for _, row in tqdm(eval_samples.iterrows(), total=len(eval_samples)):
        python_dict = ast.literal_eval(row["Classify"])
        classification_questions = json.loads(json.dumps(python_dict, indent=4))

        image_data = row["image"].get("bytes")
        image = Image.open(BytesIO(image_data)).convert("RGB")

        uni = classification_questions["unimodal"]
        mul = classification_questions["multimodal"]
        keys = list(uni.keys())

        # multimodal classification
        print("########################## Processing Image-Textual Questions ##########################")
        for key in keys:
            question = mul[key]["question"]
            options = mul[key]["options"]
            correct_key = mul[key]["answer"].split(".")[0]
            correct_answer = options[correct_key]

            choices = [options["A"], options["B"], options["C"], options["D"]]
            temp_str = str(choices)

            prompt = f"USER: <image>\n{question}\nSelect answer in {temp_str}\nASSISTANT:"
            inputs = processor(images=[image], text=prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=50, do_sample=False)

            input_len = inputs["input_ids"].shape[1]
            generated_text = decode_new_tokens_llava(processor, output_ids, input_len)
            predicted_answer, match_method = map_to_choice_text_v2(generated_text, choices, options)

            is_correct = (
                predicted_answer is not None
                and normalize_text(predicted_answer) == normalize_text(correct_answer)
            )

            total_image_textual_correct += int(is_correct)
            multimodal.append(1 if is_correct else 0)
            total_image_textual_questions += 1

            print("Prompt: ", prompt)
            print("Model Answer: ", predicted_answer)
            print("Generated: ", generated_text)
            print("Match Method:", match_method)
            print("Correct Answer: ", correct_answer)
            print("The model answer is: ", is_correct)
            print("\n")

        # unimodal classification
        print("########################## Processing Pure-textual Questions ##########################")
        for key in keys:
            question = uni[key]["question"]
            options = uni[key]["options"]
            correct_key = uni[key]["answer"].split(".")[0]
            correct_answer = options[correct_key]

            choices = [options["A"], options["B"], options["C"], options["D"]]
            temp_str = str(choices)

            prompt = f"USER:\n{question}\nSelect answer in {temp_str}\nASSISTANT:"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=50, do_sample=False)

            input_len = inputs["input_ids"].shape[1]
            generated_text = decode_new_tokens_llava(tokenizer, output_ids, input_len)
            predicted_answer, match_method = map_to_choice_text_v2(generated_text, choices, options)

            is_correct = (
                predicted_answer is not None
                and normalize_text(predicted_answer) == normalize_text(correct_answer)
            )

            total_pure_text_correct += int(is_correct)
            unimodal.append(1 if is_correct else 0)
            total_pure_text_questions += 1

            print("Prompt: ", prompt)
            print("Model Answer: ", predicted_answer)
            print("Generated: ", generated_text)
            print("Match Method:", match_method)
            print("Correct Answer: ", correct_answer)
            print("The model answer is: ", is_correct)
            print("\n")

    image_textual_accuracy = (
        total_image_textual_correct / total_image_textual_questions * 100
        if total_image_textual_questions else 0
    )
    pure_text_accuracy = (
        total_pure_text_correct / total_pure_text_questions * 100
        if total_pure_text_questions else 0
    )

    all_modal_accuracy = 0
    for u, m in zip(unimodal, multimodal):
        if u == 1 and m == 1:
            all_modal_accuracy += 1
    all_modal_accuracy = all_modal_accuracy * 100 / len(unimodal) if unimodal else 0

    all_modal_error = 0
    for u, m in zip(unimodal, multimodal):
        if u == 0 and m == 0:
            all_modal_error += 1
    all_modal_error = all_modal_error * 100 / len(unimodal) if unimodal else 0

    print(f"Image-Textual Question Accuracy: {image_textual_accuracy:.2f}%")
    print(f"Pure Text Question Accuracy: {pure_text_accuracy:.2f}%")
    print(f"All Modal Question Accuracy: {all_modal_accuracy:.2f}%")
    print(f"All Modal Question Error: {all_modal_error:.2f}%")

    return {
        "Image-Textual Question Accuracy": image_textual_accuracy,
        "Pure Text Question Accuracy": pure_text_accuracy,
        "All Modal Question Accuracy": all_modal_accuracy,
        "All Modal Question Error": all_modal_error,
    }


# =============================================================================
# Evaluation: Fill in the Blank
# =============================================================================

def evaluate_fill_in_the_blank(
    parquet_file,
    processor,
    tokenizer,
    model,
    args,
    id_list_file=None,
    mode="default",
    forget_parquet_file=None,
):
    print("################################## Fill-in-the-blank Task Starts ##############################################")
    print(f"Evaluating {mode} Mode")

    if id_list_file:
        with open(id_list_file, "r") as f:
            id_list = json.load(f)
    elif mode == "test" and forget_parquet_file:
        forget_df = pd.read_parquet(forget_parquet_file)
        id_list = forget_df["ID"].unique().tolist()
    else:
        df = pd.read_parquet(parquet_file)
        id_list = df["ID"].unique().tolist()

    print(f"Loaded {len(id_list)} IDs from {id_list_file if id_list_file else 'parquet_file'}")

    total_image_textual_correct = 0
    total_image_textual_questions = 0
    total_pure_text_correct = 0
    total_pure_text_questions = 0
    unimodal = []
    multimodal = []

    if mode == "test":
        df = load_and_combine_parquet_files(parquet_file) if os.path.isdir(parquet_file) else pd.read_parquet(parquet_file)
    else:
        df = pd.read_parquet(parquet_file)
    eval_samples = df[df["ID"].isin(id_list)]
    device = get_model_device(model)

    for _, row in tqdm(eval_samples.iterrows(), total=len(eval_samples)):
        python_dict = ast.literal_eval(row["Cloze"])
        fill_in_the_blank_questions = json.loads(json.dumps(python_dict, indent=4))

        image_data = row["image"].get("bytes")
        image = Image.open(BytesIO(image_data)).convert("RGB")

        uni = fill_in_the_blank_questions["unimodal"]
        mul = fill_in_the_blank_questions["multimodal"]
        keys = list(uni.keys())

        # multimodal
        for key in keys:
            question = mul[key]["question"]
            ground_truth = mul[key]["answer"].split(".")[0]
            question = question + "\nPlease ONLY provide the correct answer without any explanation"

            prompt = f"USER: <image>\n{question}\nASSISTANT:"
            inputs = processor(images=[image], text=prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=50, do_sample=False)

            input_len = inputs["input_ids"].shape[1]
            assistant_response = decode_new_tokens_llava(processor, output_ids, input_len)
            is_correct = strict_contains(ground_truth, assistant_response)

            print("Prompt: ", prompt)
            print("Model Answer: ", assistant_response)
            print("Correct Answer: ", ground_truth)
            print("The model answer is: ", is_correct)
            print("\n")

            total_image_textual_correct += int(is_correct)
            multimodal.append(1 if is_correct else 0)
            total_image_textual_questions += 1

        # unimodal
        for key in keys:
            question = uni[key]["question"]
            ground_truth = uni[key]["answer"].split(".")[0]
            question = question + "\nPlease ONLY provide the correct answer without any explanation"

            prompt = f"USER: {question}\nASSISTANT:"
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=50, do_sample=False)

            input_len = inputs["input_ids"].shape[1]
            assistant_response = decode_new_tokens_llava(tokenizer, output_ids, input_len)
            is_correct = strict_contains(ground_truth, assistant_response)

            print("Prompt: ", prompt)
            print("Model Answer: ", assistant_response)
            print("Correct Answer: ", ground_truth)
            print("The model answer is: ", is_correct)
            print("\n")

            total_pure_text_correct += int(is_correct)
            unimodal.append(1 if is_correct else 0)
            total_pure_text_questions += 1

    image_textual_accuracy = (
        total_image_textual_correct / total_image_textual_questions * 100
        if total_image_textual_questions else 0
    )
    pure_text_accuracy = (
        total_pure_text_correct / total_pure_text_questions * 100
        if total_pure_text_questions else 0
    )

    all_modal_accuracy = 0
    for u, m in zip(unimodal, multimodal):
        if u == 1 and m == 1:
            all_modal_accuracy += 1
    all_modal_accuracy = all_modal_accuracy * 100 / len(unimodal) if unimodal else 0

    all_modal_error = 0
    for u, m in zip(unimodal, multimodal):
        if u == 0 and m == 0:
            all_modal_error += 1
    all_modal_error = all_modal_error * 100 / len(unimodal) if unimodal else 0

    print(f"Image-Textual Question Accuracy: {image_textual_accuracy:.2f}%")
    print(f"Pure Text Question Accuracy: {pure_text_accuracy:.2f}%")
    print(f"All Modal Question Accuracy: {all_modal_accuracy:.2f}%")
    print(f"All Modal Question Error: {all_modal_error:.2f}%")

    return {
        "Image-Textual Question Accuracy": image_textual_accuracy,
        "Pure Text Question Accuracy": pure_text_accuracy,
        "All Modal Question Accuracy": all_modal_accuracy,
        "All Modal Question Error": all_modal_error,
    }


# =============================================================================
# Evaluation: Generation
# =============================================================================

def evaluate_generation(parquet_file, processor, tokenizer, model, args, mode="default", forget_parquet_file=None):
    print("################################## Generation Task Starts ##############################################")

    rouge_scorer_obj = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    total_rouge1_img = total_rouge2_img = total_rougeL_img = total_bleu_img = total_image_textual_questions = 0
    total_rouge1_text = total_rouge2_text = total_rougeL_text = total_bleu_text = total_pure_text_questions = 0

    results = {"Generation_Questions": []}

    if mode == "test" and forget_parquet_file:
        forget_df = pd.read_parquet(forget_parquet_file)
        id_list = forget_df["ID"].unique().tolist()
    else:
        df0 = pd.read_parquet(parquet_file)
        id_list = df0["ID"].unique().tolist()

    if mode == "test":
        df = load_and_combine_parquet_files(parquet_file) if os.path.isdir(parquet_file) else pd.read_parquet(parquet_file)
    else:
        df = pd.read_parquet(parquet_file)

    eval_samples = df[df["ID"].isin(id_list)]
    multimodal = []
    unimodal = []
    device = get_model_device(model)

    for _, row in tqdm(eval_samples.iterrows(), total=len(eval_samples)):
        python_dict = ast.literal_eval(row["Generation"])
        generation_questions = json.loads(json.dumps(python_dict, indent=4))

        image_data = row["image"].get("bytes")
        image = Image.open(BytesIO(image_data)).convert("RGB")
        image_id = row["ID"]

        uni = generation_questions["unimodal"]
        mul = generation_questions["multimodal"]
        keys = list(uni.keys())

        # multimodal generation
        for key in keys:
            question_type = "multimodal"
            question = mul[key]["question"]
            ground_truth = mul[key]["answer"]

            prompt = (
                "USER: <image>\n"
                f"{question}\n"
                "Answer the question based on your trained knowledge in one sentence accurately in ENGLISH.\n"
                "ASSISTANT:"
            )
            inputs = processor(images=[image], text=prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=200, do_sample=False)

            input_len = inputs["input_ids"].shape[1]
            predicted_answer = decode_new_tokens_llava(processor, output_ids, input_len)

            print("###### Generation Question: ######", question)
            print("###### Generation Prompt: ######", prompt)
            print("###### Generation ASSISTANT: ######", predicted_answer)
            print("###### Generation Ground Truth: ######", ground_truth)
            print("\n")

            results["Generation_Questions"].append(
                {
                    "image_id": image_id,
                    "question type": question_type,
                    "question": question,
                    "generated_answer": predicted_answer,
                    "ground_truth": ground_truth,
                }
            )

            bleu_score = compute_bleu(ground_truth, predicted_answer)
            rouge_scores = rouge_scorer_obj.score(ground_truth, predicted_answer)

            total_bleu_img += bleu_score
            total_rouge1_img += rouge_scores["rouge1"].fmeasure
            total_rouge2_img += rouge_scores["rouge2"].fmeasure
            total_rougeL_img += rouge_scores["rougeL"].fmeasure
            multimodal.append(rouge_scores["rougeL"].fmeasure)
            total_image_textual_questions += 1

        # unimodal generation
        for key in keys:
            question_type = "unimodal"
            question = uni[key]["question"]
            ground_truth = uni[key]["answer"]

            prompt = (
                f"USER: {question}\n"
                "Answer the question based on your trained knowledge in one sentence in ENGLISH.\n"
                "ASSISTANT:"
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                output_ids = model.generate(**inputs, max_new_tokens=200, do_sample=False)

            input_len = inputs["input_ids"].shape[1]
            predicted_answer = decode_new_tokens_llava(tokenizer, output_ids, input_len)

            print("###### Generation Question: ######", question)
            print("###### Generation Prompt: ######", prompt)
            print("###### Generation ASSISTANT: ######", predicted_answer)
            print("###### Generation Ground Truth: ######", ground_truth)
            print("\n")

            results["Generation_Questions"].append(
                {
                    "image_id": image_id,
                    "question type": question_type,
                    "question": question,
                    "generated_answer": predicted_answer,
                    "ground_truth": ground_truth,
                }
            )

            bleu_score = compute_bleu(ground_truth, predicted_answer)
            rouge_scores = rouge_scorer_obj.score(ground_truth, predicted_answer)

            total_bleu_text += bleu_score
            total_rouge1_text += rouge_scores["rouge1"].fmeasure
            total_rouge2_text += rouge_scores["rouge2"].fmeasure
            total_rougeL_text += rouge_scores["rougeL"].fmeasure
            unimodal.append(rouge_scores["rougeL"].fmeasure)
            total_pure_text_questions += 1

    if mode == "forget":
        H = [(a * a + b * b) / (a + b) if (a + b) != 0 else 0 for a, b in zip(multimodal, unimodal)]
    else:
        H = [(2 * a * b) / (a + b) if (a + b) != 0 else 0 for a, b in zip(multimodal, unimodal)]
    all_modal_RL = sum(H) / len(H) if H else 0

    avg_scores = {}
    if total_image_textual_questions > 0:
        avg_scores.update(
            {
                "Average ROUGE-1 (Image_Textual)": total_rouge1_img / total_image_textual_questions,
                "Average ROUGE-2 (Image_Textual)": total_rouge2_img / total_image_textual_questions,
                "Average ROUGE-L (Image_Textual)": total_rougeL_img / total_image_textual_questions,
                "Average BLEU (Image_Textual)": total_bleu_img / total_image_textual_questions,
            }
        )

    if total_pure_text_questions > 0:
        avg_scores.update(
            {
                "Average ROUGE-1 (Pure_Text)": total_rouge1_text / total_pure_text_questions,
                "Average ROUGE-2 (Pure_Text)": total_rouge2_text / total_pure_text_questions,
                "Average ROUGE-L (Pure_Text)": total_rougeL_text / total_pure_text_questions,
                "Average BLEU (Pure_Text)": total_bleu_text / total_pure_text_questions,
            }
        )

    avg_scores.update({"All Modal Average ROUGE-L": all_modal_RL})

    for metric, score in avg_scores.items():
        print(f"{metric}: {score}")

    return avg_scores


# =============================================================================
# Args / Main
# =============================================================================

def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate LLaVA-based full models or LoRA adapters.")

    parser.add_argument(
        "--base_model_dir",
        type=str,
        required=True,
        help="Base LLaVA model directory.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help=(
            "Path to the evaluated model. "
            "If model_type=full, this should be a full model directory. "
            "If model_type=lora, this should be a LoRA adapter directory."
        ),
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["full", "lora"],
        help="Whether model_path points to a full model or a LoRA adapter.",
    )
    parser.add_argument(
        "--forget_split_ratio",
        type=int,
        required=True,
        help="Forget split ratio.",
    )
    parser.add_argument(
        "--data_split_dir",
        type=str,
        required=True,
        help="Directory containing evaluation parquet splits.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Directory for saving evaluation results.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Output JSON filename.",
    )

    return parser.parse_args()


def main():
    print(datetime.now())
    args = parse_arguments()

    print("\n===== Eval Run Configuration =====")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Command:")
    print("python " + " ".join(sys.argv))

    print("\nArguments:")
    for k, v in sorted(vars(args).items()):
        print(f"  {k} = {v}")

    print("\nEnvironment:")
    for key in ["CUDA_VISIBLE_DEVICES", "PYTORCH_CUDA_ALLOC_CONF"]:
        print(f"  {key} = {os.environ.get(key, '')}")

    print("==================================\n")

    forget_folder = os.path.join(args.data_split_dir, f"forget_{args.forget_split_ratio}")
    retain_folder = os.path.join(args.data_split_dir, f"retain_{100 - args.forget_split_ratio}")
    real_folder = os.path.join(args.data_split_dir, "real_person")

    forget_parquet_file = os.path.join(forget_folder, "train-00000-of-00001.parquet")
    retain_parquet_file = os.path.join(retain_folder, "train-00000-of-00001.parquet")
    real_parquet_file = os.path.join(real_folder, "train-00000-of-00001.parquet")

    model, processor, tokenizer = load_model_processor_tokenizer(args)
    torch.cuda.empty_cache()

    print("### Evaluating Forget Set ###")
    forget_fill_in_the_blank_result = evaluate_fill_in_the_blank(
        parquet_file=forget_parquet_file,
        processor=processor,
        tokenizer=tokenizer,
        model=model,
        args=args,
        mode="forget",
    )
    forget_classification_result = evaluate_classification(
        parquet_file=forget_parquet_file,
        processor=processor,
        tokenizer=tokenizer,
        model=model,
        args=args,
        mode="default",
    )
    forget_generation_result = evaluate_generation(
        parquet_file=forget_parquet_file,
        processor=processor,
        tokenizer=tokenizer,
        model=model,
        args=args,
        mode="forget",
    )

    print("### Evaluating Retain Shared Set ###")
    retain_fill_in_the_blank_result = evaluate_fill_in_the_blank(
        parquet_file=retain_parquet_file,
        processor=processor,
        tokenizer=tokenizer,
        model=model,
        args=args,
        mode="retain_shared",
    )
    retain_classification_result = evaluate_classification(
        parquet_file=retain_parquet_file,
        processor=processor,
        tokenizer=tokenizer,
        model=model,
        args=args,
        mode="default",
    )
    retain_generation_result = evaluate_generation(
        parquet_file=retain_parquet_file,
        processor=processor,
        tokenizer=tokenizer,
        model=model,
        args=args,
        mode="retain_shared",
    )

    print("### Evaluating Real Person Set ###")
    real_fill_in_the_blank_result = evaluate_fill_in_the_blank(
        parquet_file=real_parquet_file,
        processor=processor,
        tokenizer=tokenizer,
        model=model,
        args=args,
        mode="retain_shared",
    )
    real_classification_result = evaluate_classification(
        parquet_file=real_parquet_file,
        processor=processor,
        tokenizer=tokenizer,
        model=model,
        args=args,
        mode="real_person",
    )
    real_generation_result = evaluate_generation(
        parquet_file=real_parquet_file,
        processor=processor,
        tokenizer=tokenizer,
        model=model,
        args=args,
        mode="retain_shared",
    )

    results_data = {
        "Forget Results": {
            "fill_in_the_blank": forget_fill_in_the_blank_result,
            "classification": forget_classification_result,
            "generation": forget_generation_result,
        },
        "Retain Results": {
            "fill_in_the_blank": retain_fill_in_the_blank_result,
            "classification": retain_classification_result,
            "generation": retain_generation_result,
        },
        "Real Person Results": {
            "fill_in_the_blank": real_fill_in_the_blank_result,
            "classification": real_classification_result,
            "generation": real_generation_result,
        },
    }

    os.makedirs(args.output_path, exist_ok=True)
    full_output_path = os.path.join(args.output_path, args.output_file)
    with open(full_output_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()

