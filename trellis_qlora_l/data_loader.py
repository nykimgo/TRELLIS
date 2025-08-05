"""
데이터 로더 유틸리티

주요 기능:
- TRELLIS 훈련용 데이터셋 로드
- 분산 훈련용 DistributedSampler 지원
- 텍스트-3D 쌍 데이터 전처리
"""

import os
import json
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import random

# TRELLIS 관련 임포트
try:
    from trellis.utils import preprocess_text
except ImportError:
    def preprocess_text(text):
        return text.strip().lower()


class TRELLISDataset(Dataset):
    """TRELLIS 훈련용 데이터셋"""
    
    def __init__(self, data_path: str, max_seq_length: int = 2048):
        """
        Args:
            data_path: 데이터셋 경로 (JSON 파일 또는 디렉토리)
            max_seq_length: 최대 시퀀스 길이
        """
        self.data_path = Path(data_path)
        self.max_seq_length = max_seq_length
        self.data = self._load_data()
        
        print(f"📊 데이터셋 로드 완료: {len(self.data)} 샘플")
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """데이터 로드"""
        if not self.data_path.exists():
            # 더미 데이터 생성 (실제 데이터가 없을 때)
            print(f"⚠️ 데이터 경로가 존재하지 않습니다: {self.data_path}")
            print("📊 더미 데이터로 훈련을 진행합니다")
            return self._create_dummy_data()
        
        if self.data_path.is_file() and self.data_path.suffix == '.json':
            # JSON 파일에서 로드
            with open(self.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data if isinstance(data, list) else [data]
        
        elif self.data_path.is_dir():
            # 디렉토리에서 여러 JSON 파일 로드
            data = []
            for json_file in self.data_path.glob("*.json"):
                with open(json_file, 'r', encoding='utf-8') as f:
                    file_data = json.load(f)
                    if isinstance(file_data, list):
                        data.extend(file_data)
                    else:
                        data.append(file_data)
            return data
        
        else:
            print(f"⚠️ 지원되지 않는 데이터 형식: {self.data_path}")
            return self._create_dummy_data()
    
    def _create_dummy_data(self) -> List[Dict[str, Any]]:
        """더미 데이터 생성 (테스트용)"""
        dummy_prompts = [
            "a red sports car",
            "a wooden chair with leather cushion",
            "a blue coffee mug on a table",
            "a small house with garden",
            "a cute cat sitting on a window",
            "a modern desk lamp",
            "a vintage bicycle",
            "a colorful umbrella",
            "a glass vase with flowers",
            "a leather backpack",
            "a digital camera",
            "a pair of running shoes",
            "a stack of books",
            "a ceramic bowl",
            "a metal water bottle"
        ]
        
        data = []
        for i in range(100):  # 100개 더미 샘플
            prompt = random.choice(dummy_prompts)
            data.append({
                "text": prompt,
                "id": f"dummy_{i:03d}",
                # 실제로는 3D 관련 데이터 (좌표, 메시 등)가 포함됨
                "target": self._create_dummy_3d_data()
            })
        
        return data
    
    def _create_dummy_3d_data(self) -> Dict[str, Any]:
        """더미 3D 데이터 생성"""
        # 실제로는 TRELLIS의 구조화된 3D 표현이 사용됨
        return {
            "structure": torch.randn(64, 64, 64),  # 구조 표현
            "appearance": torch.randn(128, 128, 3),  # 외형 표현
            "geometry": torch.randn(1000, 3)  # 기하 정보
        }
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """데이터 샘플 반환"""
        item = self.data[idx]
        
        # 텍스트 전처리
        text = preprocess_text(item["text"])
        
        # 토큰화 (실제로는 TRELLIS의 텍스트 인코더 사용)
        # 여기서는 단순화된 버전 사용
        tokens = self._tokenize_text(text)
        
        # 3D 타겟 데이터
        target = item.get("target", self._create_dummy_3d_data())
        
        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "labels": target,  # 3D 생성 타겟
            "text": text  # 원본 텍스트 (디버깅용)
        }
    
    def _tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        """텍스트 토큰화 (간단한 더미 구현)"""
        # 실제로는 TRELLIS의 텍스트 인코더를 사용해야 함
        words = text.split()
        
        # 간단한 단어 -> 인덱스 매핑
        vocab = {
            "<pad>": 0, "<unk>": 1, "a": 2, "red": 3, "car": 4, 
            "blue": 5, "chair": 6, "house": 7, "cat": 8, "dog": 9,
            "table": 10, "book": 11, "flower": 12, "tree": 13, "sky": 14
        }
        
        # 토큰 ID 생성
        token_ids = []
        for word in words[:self.max_seq_length-2]:  # 패딩 고려
            token_ids.append(vocab.get(word.lower(), 1))  # <unk>
        
        # 패딩
        seq_len = min(len(token_ids), self.max_seq_length)
        input_ids = token_ids[:seq_len] + [0] * (self.max_seq_length - seq_len)
        attention_mask = [1] * seq_len + [0] * (self.max_seq_length - seq_len)
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
        }


def create_dataloader(config, world_size: int = 1, rank: int = 0) -> Tuple[DataLoader, Optional[DataLoader]]:
    """데이터로더 생성"""
    
    # 데이터셋 타입에 따른 로더 선택
    if hasattr(config, 'dataset_type') and config.dataset_type == "hssd":
        # HSSD 데이터셋 사용
        return create_hssd_dataloader(config.dataset_path, config, world_size, rank)
    
    elif config.dataset_path and config.dataset_path != "":
        # 일반 데이터셋
        dataset = TRELLISDataset(
            data_path=config.dataset_path,
            max_seq_length=config.max_seq_length
        )
    else:
        # 더미 데이터셋 (테스트용)
        dataset = TRELLISDataset(
            data_path="dummy",
            max_seq_length=config.max_seq_length
        )
    
    # 훈련/검증 분할
    if config.dataset_split_ratio < 1.0:
        train_size = int(len(dataset) * config.dataset_split_ratio)
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
    else:
        train_dataset = dataset
        val_dataset = None
    
    # 분산 샘플러
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset, 
            num_replicas=world_size, 
            rank=rank,
            shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset, 
            num_replicas=world_size, 
            rank=rank,
            shuffle=False
        ) if val_dataset else None
    else:
        train_sampler = None
        val_sampler = None
    
    # 훈련 데이터로더
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size_per_gpu,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=config.dataloader_num_workers,
        pin_memory=config.dataloader_pin_memory,
        drop_last=True,
        collate_fn=collate_fn
    )
    
    # 검증 데이터로더
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size_per_gpu,
            sampler=val_sampler,
            shuffle=False,
            num_workers=config.dataloader_num_workers,
            pin_memory=config.dataloader_pin_memory,
            drop_last=False,
            collate_fn=collate_fn
        )
    
    return train_loader, val_loader


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """배치 collate 함수"""
    
    # 텍스트 관련 데이터
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    
    # 3D 라벨 데이터 (간단히 구조만 맞춤)
    labels = {}
    if "labels" in batch[0]:
        for key in batch[0]["labels"]:
            if isinstance(batch[0]["labels"][key], torch.Tensor):
                labels[key] = torch.stack([item["labels"][key] for item in batch])
    
    # 원본 텍스트 (디버깅용)
    texts = [item["text"] for item in batch]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "texts": texts
    }


class HSSDDataset(TRELLISDataset):
    """HSSD 데이터셋 전용 클래스"""
    
    def __init__(self, hssd_path: str, split: str = "train", **kwargs):
        """
        Args:
            hssd_path: HSSD 데이터셋 루트 경로
            split: 데이터 분할 (train/val/test)
        """
        self.hssd_path = Path(hssd_path)
        self.split = split
        
        # HSSD 데이터 구조 확인
        self.expected_dirs = [
            "metadata",    # 메타데이터
            "rendered",    # 렌더링된 이미지  
            "voxelized",   # 복셀 데이터
            "latents"      # 인코딩된 latent
        ]
        
        super().__init__(hssd_path, **kwargs)
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """HSSD 데이터 로드"""
        
        # HSSD 구조 확인
        if not self._validate_hssd_structure():
            print(f"⚠️ HSSD 구조가 올바르지 않습니다: {self.hssd_path}")
            return self._create_dummy_data()
        
        # 메타데이터 파일들 찾기
        metadata_files = []
        metadata_dir = self.hssd_path / "metadata"
        
        if metadata_dir.exists():
            # split별 메타데이터 파일 찾기
            for pattern in [f"{self.split}.json", f"{self.split}_*.json", "*.json"]:
                files = list(metadata_dir.glob(pattern))
                metadata_files.extend(files)
        
        if not metadata_files:
            print(f"⚠️ HSSD 메타데이터를 찾을 수 없습니다: {metadata_dir}")
            return self._create_dummy_data()
        
        print(f"📊 HSSD 메타데이터 파일: {len(metadata_files)}개")
        
        # 메타데이터 로드
        data = []
        for meta_file in metadata_files:
            try:
                with open(meta_file, 'r') as f:
                    file_data = json.load(f)
                
                if isinstance(file_data, list):
                    for item in file_data:
                        processed_item = self._process_hssd_item(item)
                        if processed_item:
                            data.append(processed_item)
                else:
                    processed_item = self._process_hssd_item(file_data)
                    if processed_item:
                        data.append(processed_item)
                        
            except Exception as e:
                print(f"⚠️ 메타데이터 파일 로드 실패: {meta_file} - {e}")
                continue
        
        print(f"📊 HSSD 데이터 로드 완료: {len(data)}개 샘플")
        return data
    
    def _validate_hssd_structure(self) -> bool:
        """HSSD 데이터 구조 검증"""
        if not self.hssd_path.exists():
            return False
        
        missing_dirs = []
        for dir_name in self.expected_dirs:
            if not (self.hssd_path / dir_name).exists():
                missing_dirs.append(dir_name)
        
        if missing_dirs:
            print(f"⚠️ HSSD 디렉토리 누락: {missing_dirs}")
            # metadata만 있어도 기본 동작은 가능
            return (self.hssd_path / "metadata").exists()
        
        return True
    
    def _process_hssd_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """HSSD 항목 처리"""
        try:
            # 필수 필드 확인
            if 'object_id' not in item:
                return None
            
            object_id = item['object_id']
            
            # 텍스트 설명 추출
            text_description = self._extract_text_description(item)
            if not text_description:
                return None
            
            # 3D 데이터 경로들
            data_paths = self._get_hssd_data_paths(object_id)
            
            return {
                "text": text_description,
                "id": object_id,
                "target": data_paths,
                "metadata": item
            }
            
        except Exception as e:
            print(f"⚠️ HSSD 항목 처리 실패: {e}")
            return None
    
    def _extract_text_description(self, item: Dict[str, Any]) -> str:
        """텍스트 설명 추출"""
        # 다양한 필드에서 텍스트 찾기
        text_fields = ['description', 'caption', 'text', 'prompt', 'name', 'category']
        
        for field in text_fields:
            if field in item and item[field]:
                return str(item[field]).strip()
        
        # 카테고리나 클래스에서 기본 텍스트 생성
        if 'category' in item:
            return f"a {item['category']}"
        elif 'class' in item:
            return f"a {item['class']}"
        
        return None
    
    def _get_hssd_data_paths(self, object_id: str) -> Dict[str, Any]:
        """HSSD 3D 데이터 경로들 반환"""
        paths = {}
        
        # 렌더링된 이미지
        rendered_dir = self.hssd_path / "rendered" / object_id
        if rendered_dir.exists():
            paths["rendered_images"] = list(rendered_dir.glob("*.png"))
        
        # 복셀 데이터
        voxel_file = self.hssd_path / "voxelized" / f"{object_id}.npz"
        if voxel_file.exists():
            paths["voxel_data"] = voxel_file
        
        # 인코딩된 latent
        latent_file = self.hssd_path / "latents" / f"{object_id}.pt"
        if latent_file.exists():
            paths["latent_data"] = latent_file
        
        # 메시 파일
        mesh_file = self.hssd_path / "meshes" / f"{object_id}.ply"
        if mesh_file.exists():
            paths["mesh_data"] = mesh_file
        
        return paths
    
    def _load_hssd_3d_data(self, data_paths: Dict[str, Any]) -> Dict[str, Any]:
        """실제 HSSD 3D 데이터 로드"""
        loaded_data = {}
        
        try:
            # 복셀 데이터 로드
            if "voxel_data" in data_paths:
                voxel_data = np.load(data_paths["voxel_data"])
                loaded_data["voxels"] = torch.from_numpy(voxel_data['voxels'])
            
            # Latent 데이터 로드
            if "latent_data" in data_paths:
                latent_data = torch.load(data_paths["latent_data"], map_location='cpu')
                loaded_data["latents"] = latent_data
            
            # 렌더링 이미지 (첫 번째만)
            if "rendered_images" in data_paths and data_paths["rendered_images"]:
                # 실제로는 이미지를 로드하지만, 여기서는 경로만 저장
                loaded_data["rendered_path"] = str(data_paths["rendered_images"][0])
                
        except Exception as e:
            print(f"⚠️ HSSD 3D 데이터 로드 실패: {e}")
            # 실패 시 더미 데이터
            loaded_data = self._create_dummy_3d_data()
        
        return loaded_data
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """데이터 샘플 반환 (HSSD 버전)"""
        item = self.data[idx]
        
        # 텍스트 전처리
        text = preprocess_text(item["text"])
        tokens = self._tokenize_text(text)
        
        # 3D 데이터 로드
        if isinstance(item["target"], dict) and "latent_data" in item["target"]:
            # 실제 HSSD 데이터 로드
            target_data = self._load_hssd_3d_data(item["target"])
        else:
            # 더미 데이터
            target_data = self._create_dummy_3d_data()
        
        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "labels": target_data,
            "text": text,
            "object_id": item["id"],
            "metadata": item.get("metadata", {})
        }


def create_hssd_dataloader(hssd_path: str, config, world_size: int = 1, rank: int = 0):
    """HSSD 데이터로더 생성"""
    
    # 훈련/검증 데이터셋
    train_dataset = HSSDDataset(hssd_path, split="train", max_seq_length=config.max_seq_length)
    val_dataset = HSSDDataset(hssd_path, split="val", max_seq_length=config.max_seq_length)
    
    # 분산 샘플러
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
    else:
        train_sampler = None
        val_sampler = None
    
    # 데이터로더
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size_per_gpu,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=config.dataloader_num_workers,
        pin_memory=config.dataloader_pin_memory,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size_per_gpu,
        sampler=val_sampler,
        shuffle=False,
        num_workers=config.dataloader_num_workers,
        pin_memory=config.dataloader_pin_memory,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader