import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

IMAGE_CAPTION_QUESTIONS = [
    "What can you see in this picture?",
    "Tell me about the content of this image",
    "Can you give a description of the image?",
    "What is depicted in the image?",
    "Explain what you observe in the picture.",
    "Describe the image in detail.",
    "What is the main subject of this image?",
    "Can you describe the scene or objects in the image?",
    "What is happening in this image?",
]

TEXT_QA_MODE = "TEXT_QA"
IMAGE_QA_MODE = "IMAGE_QA"
CAPTION_MODE = "CAPTION"
RECOGNITION_MODE = "RECOGNITION"


class CLEARDataset(Dataset):
    """
    CLEAR 数据统一映射成三元组：
        {
            "image": ... or None,
            "question": str,
            "answer": str
        }

    这里把不同训练子任务拆成显式模式，避免模式之间隐式重叠。
    """

    def __init__(self, data, mode: str):
        super().__init__()
        self.data = data
        self.mode = mode
        self.samples = self._build_samples()

    def _build_samples(self):
        samples = []

        for item in self.data:
            image = item.get("image", None)
            question = item.get("question", "")
            answer = item.get("answer", "")
            caption = item.get("caption", "")
            name = item.get("name", "")

            # 1) 纯文本 QA
            if self.mode == TEXT_QA_MODE:
                if image is None and question and answer:
                    samples.append({
                        "image": None,
                        "question": question,
                        "answer": answer,
                    })
                continue

            # 2) 原始图文 QA
            if self.mode == IMAGE_QA_MODE:
                if image is not None and question and answer:
                    samples.append({
                        "image": image,
                        "question": question,
                        "answer": answer,
                    })
                continue

            # 3) 图像描述任务，构造成 VQA 风格
            if self.mode == CAPTION_MODE:
                if image is not None and caption:
                    samples.append({
                        "image": image,
                        "question": random.choice(IMAGE_CAPTION_QUESTIONS),
                        "answer": caption,
                    })
                continue

            # 4) 人脸/人物识别名任务（当前训练脚本没默认启用）
            if self.mode == RECOGNITION_MODE:
                if image is not None and name:
                    samples.append({
                        "image": image,
                        "question": "The name of the person in the image is",
                        "answer": name,
                    })
                continue

            raise ValueError(f"Unsupported mode: {self.mode}")

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


@dataclass
class ClearCollator:
    """
    负责：
    1. 把样本组装成 chat template
    2. 做 batch processor
    3. 生成 labels
    4. 在 ans_only 模式下只监督 assistant answer 区域
    5. debug 打印
    """
    processor: object
    ans_only: bool = False
    debug: bool = False
    debug_max_prints: int = 2

    def __post_init__(self):
        self._debug_print_count = 0

    def _build_messages(self, sample):
        image = sample.get("image")
        question = sample.get("question")
        answer = sample.get("answer")

        if image is None:
            user_content = [{"type": "text", "text": question}]
        else:
            user_content = [
                {"type": "image"},
                {"type": "text", "text": question},
            ]

        prompt_messages = [
            {
                "role": "user",
                "content": user_content,
            }
        ]

        full_messages = [
            {
                "role": "user",
                "content": user_content,
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            },
        ]

        return image, prompt_messages, full_messages

    def _debug_print(self, batch, texts, prompt_lengths=None):
        if not self.debug or self._debug_print_count >= self.debug_max_prints:
            return

        idx = 0
        input_ids = batch["input_ids"][idx]
        labels = batch["labels"][idx]

        decoded_full = self.processor.tokenizer.decode(input_ids, skip_special_tokens=False)

        supervised_ids = input_ids.clone()
        supervised_ids[labels == -100] = self.processor.tokenizer.pad_token_id
        decoded_supervised = self.processor.tokenizer.decode(
            supervised_ids, skip_special_tokens=False
        )

        print("\n" + "=" * 100)
        print(f"[DEBUG SAMPLE #{self._debug_print_count + 1}]")
        print(f"Full text after chat template:\n{texts[idx]}")
        print("-" * 100)
        print(f"Decoded full input_ids:\n{decoded_full}")
        print("-" * 100)
        if prompt_lengths is not None:
            print(f"Prompt length (tokens): {prompt_lengths[idx]}")
        print(f"Supervised region after masking:\n{decoded_supervised}")
        print("=" * 100 + "\n")

        self._debug_print_count += 1

    def __call__(self, examples):
        texts = []
        images = []
        prompt_lengths = []

        for sample in examples:
            image, prompt_messages, full_messages = self._build_messages(sample)

            full_text = self.processor.apply_chat_template(
                full_messages,
                add_generation_prompt=False,
            ).strip()
            texts.append(full_text)

            if image is not None:
                images.append(image)

            if self.ans_only:
                prompt_text = self.processor.apply_chat_template(
                    prompt_messages,
                    add_generation_prompt=True,
                ).strip()

                prompt_inputs = self.processor(
                    text=[prompt_text],
                    images=[image] if image is not None else None,
                    padding=False,
                    truncation=True,
                    return_tensors="pt",
                )
                prompt_lengths.append(prompt_inputs["input_ids"].shape[1])

        batch = self.processor(
            text=texts,
            images=images if len(images) > 0 else None,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()

        if self.ans_only:
            for i, prompt_len in enumerate(prompt_lengths):
                labels[i, :prompt_len] = -100

        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        self._debug_print(batch=batch, texts=texts, prompt_lengths=prompt_lengths if self.ans_only else None)
        return batch
