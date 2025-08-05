"""간단한 텍스트 데이터셋 로더"""

from pathlib import Path
from typing import List

import json
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """HSSD와 같은 JSONL 형식 텍스트 데이터셋 로더"""

    def __init__(self, data_dir: str, tokenizer, max_length: int = 256) -> None:
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.files: List[Path] = sorted(self.data_dir.glob("*.json"))
        if not self.files:
            raise FileNotFoundError(f"데이터셋을 찾을 수 없습니다: {data_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        file_path = self.files[idx]
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        text = data.get("text", "")
        tokens = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
        item = {k: v.squeeze(0) for k, v in tokens.items()}
        item["labels"] = item["input_ids"].clone()
        return item
