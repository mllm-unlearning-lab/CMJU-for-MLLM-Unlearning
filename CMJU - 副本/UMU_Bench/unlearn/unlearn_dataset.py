import ast
import json
from io import BytesIO
from typing import Any

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


def _find_assistant_start(input_ids_tensor, tokenizer):
    ids = input_ids_tensor.tolist()
    n = len(ids)
    target = "ASSISTANT:"

    if target not in tokenizer.decode(ids, skip_special_tokens=False):
        return None

    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        decoded = tokenizer.decode(ids[:mid], skip_special_tokens=False)
        if target in decoded:
            hi = mid
        else:
            lo = mid + 1
    return lo


class Multimodal_Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, sort_json_key: bool = True):
        super().__init__()
        self.df = df
        self.sort_json_key = sort_json_key
        self.dataset = self.flatten_dataset()

    def flatten_dataset(self):
        flattened_data = []

        for idx, row in self.df.iterrows():
            image_data = row["image"].get("bytes")
            try:
                image = Image.open(BytesIO(image_data)).convert("RGB")
            except Exception as e:
                print(f"Error loading image at index {idx}: {e}")
                continue

            qa_dict = ast.literal_eval(row["MM_QA"])
            qas = json.loads(json.dumps(qa_dict, indent=4))
            questions = qas["question"]
            answers = qas["answer"]

            for k in questions.keys():
                flattened_data.append({
                    "image": image,
                    "question": questions[k],
                    "answer": answers[k],
                })

        return flattened_data

    def __len__(self):
        return len(self.dataset)

    def json2token(self, obj: Any, sort_json_key: bool = True):
        if isinstance(obj, dict):
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]

            output = ""
            keys = sorted(obj.keys(), reverse=True) if sort_json_key else obj.keys()
            for k in keys:
                output += f"<s_{k}>" + self.json2token(obj[k], sort_json_key) + f"</s_{k}>"
            return output

        if isinstance(obj, list):
            return "<sep/>".join([self.json2token(item, sort_json_key) for item in obj])

        return str(obj)

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]
        question = sample.get("question", "")
        answer = sample.get("answer", "")

        tokenized_question = self.json2token(question, sort_json_key=self.sort_json_key)
        tokenized_answer = self.json2token(answer, sort_json_key=self.sort_json_key)

        return {
            "image": sample["image"],
            "question": tokenized_question,
            "answer": tokenized_answer,
        }


class Unimodal_Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, sort_json_key: bool = True):
        super().__init__()
        self.df = df
        self.sort_json_key = sort_json_key
        self.dataset = self.flatten_dataset()

    def flatten_dataset(self):
        flattened_data = []

        for row in self.df.itertuples(index=False):
            qa_dict = ast.literal_eval(row.UM_QA)
            qas = json.loads(json.dumps(qa_dict, indent=4))
            questions = qas["question"]
            answers = qas["answer"]

            for k in questions.keys():
                flattened_data.append({
                    "image": None,
                    "question": questions[k],
                    "answer": answers[k],
                })

        return flattened_data

    def __len__(self):
        return len(self.dataset)

    def json2token(self, obj: Any, sort_json_key: bool = True):
        if isinstance(obj, dict):
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]

            output = ""
            keys = sorted(obj.keys(), reverse=True) if sort_json_key else obj.keys()
            for k in keys:
                output += f"<s_{k}>" + self.json2token(obj[k], sort_json_key) + f"</s_{k}>"
            return output

        if isinstance(obj, list):
            return "<sep/>".join([self.json2token(item, sort_json_key) for item in obj])

        return str(obj)

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]
        question = sample.get("question", "")
        answer = sample.get("answer", "")

        tokenized_question = self.json2token(question, sort_json_key=self.sort_json_key)
        tokenized_answer = self.json2token(answer, sort_json_key=self.sort_json_key)

        return {
            "image": None,
            "question": tokenized_question,
            "answer": tokenized_answer,
        }


def train_collate_fn_llava_multimodal(examples, processor, args):
    images, texts = [], []
    for example in examples:
        images.append(example.get("image"))
        question = example.get("question")
        answer = example.get("answer")
        texts.append(f"USER: <image>\n{question}\nASSISTANT: {answer}")

    batch = processor(
        text=texts,
        images=images,
        padding=True,
        truncation=True,
        # max_length=args.max_length,
        return_tensors="pt",
    )

    input_ids = batch["input_ids"]
    labels = input_ids.clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    for i in range(labels.shape[0]):
        pos = _find_assistant_start(input_ids[i], processor.tokenizer)
        if pos is not None:
            labels[i, :pos] = -100
        else:
            labels[i, :] = -100
            print(f"[Warning] sample {i} missing ASSISTANT:, masked all tokens")

    batch["labels"] = labels
    return batch["input_ids"], batch["attention_mask"], batch["pixel_values"], batch["labels"]


def train_collate_fn_llava_unimodal(examples, processor, args):
    texts = []
    for example in examples:
        question = example.get("question")
        answer = example.get("answer")
        texts.append(f"USER: {question}\nASSISTANT: {answer}")

    batch = processor(
        text=texts,
        padding=True,
        truncation=True,
        # max_length=args.max_length,
        return_tensors="pt",
    )

    input_ids = batch["input_ids"]
    labels = input_ids.clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100

    for i in range(labels.shape[0]):
        pos = _find_assistant_start(input_ids[i], processor.tokenizer)
        if pos is not None:
            labels[i, :pos] = -100
        else:
            labels[i, :] = -100
            print(f"[Warning] sample {i} missing ASSISTANT:, masked all tokens")

    batch["labels"] = labels
    return batch["input_ids"], batch["attention_mask"], None, batch["labels"]
