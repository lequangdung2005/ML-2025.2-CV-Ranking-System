"""
Fine-tune Donut (Document Understanding Transformer) for CV/Resume extraction.

This script fine-tunes the Donut model on resume NER datasets to improve
OCR and information extraction capabilities using highly RAM-optimized Lazy Loading.
"""

import os
import json
from pathlib import Path
from typing import Dict, Tuple
from dataclasses import dataclass, field

import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    VisionEncoderDecoderModel,
    DonutProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
)
from datasets import load_dataset

# Import utility (Giữ nguyên theo cấu trúc thư mục của bạn)
from cv_ranking_system.utils.logging_utils import setup_logger
from cv_ranking_system.extraction.generate_synthetic_donut import SyntheticCVGenerator

os.environ["HF_HOME"] = "D:/ML/huggingface_cache"
print("Hugging Face cache directory set to:", os.environ["HF_HOME"])

logger = setup_logger(__name__)


@dataclass
class DonutConfig:
    """Configuration for Donut fine-tuning."""
    model_name: str = "naver-clova-ix/donut-base"
    output_dir: str = "./models/donut-finetuned"
    num_epochs: int = 10
    batch_size: int = 32 
    learning_rate: float = 2e-5
    max_seq_length: int = 1536
    image_size: Tuple[int, int] = field(default_factory=lambda: (1280, 960))
    warmup_steps: int = 500
    weight_decay: float = 0.01
    validation_split: float = 0.1
    seed: int = 42
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    
    # Data sources (Ưu tiên dùng dữ liệu giả lập local để tối ưu)
    use_synthetic_dataset: bool = True
    use_kaggle_dataset: bool = False
    use_huggingface_dataset: bool = False
    local_data_path: str = "./data/donut_dataset"
    auto_generate_synthetic_data: bool = True
    regenerate_synthetic_data: bool = False
    synthetic_source_dataset: str = "cnamuangtoun/resume-job-description-fit"
    synthetic_max_samples: int = 100
    
    # Training options
    gradient_accumulation_steps: int = 1 
    max_steps: int = -1
    save_steps: int = 500
    eval_steps: int = 250
    early_stopping_patience: int = 3


class ResumeDocumentDataset(Dataset):
    """Dataset tối ưu RAM: Chỉ load ảnh từ đĩa khi DataLoader yêu cầu (Lazy Loading)"""
    
    def __init__(
        self,
        hf_dataset,
        processor: DonutProcessor,
        max_seq_length: int = 1536,
    ):
        self.hf_dataset = hf_dataset
        self.processor = processor
        self.max_seq_length = max_seq_length
    
    def __len__(self) -> int:
        return len(self.hf_dataset)
    
    def __getitem__(self, idx: int) -> Dict:
        """Lấy 1 sample: Lúc này ảnh mới thực sự được tải vào RAM"""
        sample = self.hf_dataset[idx]
        
        # 1. Xử lý ảnh đầu vào cho Encoder
        image = sample["image"].convert("RGB")
        pixel_values = self.processor(
            image, 
            return_tensors="pt"
        ).pixel_values.squeeze(0)
        
        # 2. Xử lý văn bản đích (Target) cho Decoder
        text = sample["ground_truth"]
        
        # Bọc token nhiệm vụ
        if "<s_resume>" not in text:
            full_text = f"<s_resume>{text}</s_resume>" + self.processor.tokenizer.eos_token
        else:
            full_text = text + self.processor.tokenizer.eos_token
        
        # Tokenize chuỗi đích
        target_tokenizer_output = self.processor.tokenizer(
            full_text,
            max_length=self.max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        labels = target_tokenizer_output["input_ids"].squeeze(0)
        
        # QUAN TRỌNG: Thay thế ID của pad_token bằng -100 để không tính Loss
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        return {
            "pixel_values": pixel_values,
            "labels": labels
        }


class DonutFineTuner:
    """Handler for Donut model fine-tuning."""
    
    def __init__(self, config: DonutConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        logger.info(f"Initializing Donut fine-tuner on {self.device}")
        logger.info(f"Config: {config}")

    def _split_has_samples(self, split_dir: Path) -> bool:
        """Return True when an imagefolder split has metadata and image files."""
        metadata_file = split_dir / "metadata.jsonl"
        if not metadata_file.exists() or metadata_file.stat().st_size == 0:
            return False

        image_extensions = {".png", ".jpg", ".jpeg", ".webp"}
        return any(
            path.is_file() and path.suffix.lower() in image_extensions
            for path in split_dir.iterdir()
        )

    def _synthetic_dataset_ready(self) -> bool:
        data_dir = Path(self.config.local_data_path)
        return self._split_has_samples(data_dir / "train")

    def ensure_synthetic_dataset(self):
        """Tự động sinh dataset nếu chưa tồn tại cục bộ."""
        local_path = Path(self.config.local_data_path)
        
        if not (local_path / "train" / "metadata.jsonl").exists():
            logger.info("Dataset không tồn tại cục bộ. Đang tự động sinh dữ liệu từ file NER...")
            
            project_root = Path(__file__).resolve().parents[3]
            
            ner_data_path = project_root / "data" / "resume_ner" / "train.json"
            
            logger.info(f"Đang đọc dữ liệu NER từ đường dẫn động: {ner_data_path}")
            
            if not ner_data_path.exists():
                raise FileNotFoundError(
                    f"Không tìm thấy file data tại {ner_data_path}. "
                    f"Vui lòng đảm bảo cấu trúc thư mục data/resume_ner/train.json là chính xác."
                )
            
            generator = SyntheticCVGenerator(output_dir=self.config.local_data_path)
            generator.process_and_convert(
                kaggle_json_path=str(ner_data_path),
                max_samples=100
            )
    
    def load_model_and_processor(self) -> Tuple[VisionEncoderDecoderModel, DonutProcessor]:
        # ... 
        processor = DonutProcessor.from_pretrained(self.config.model_name)
        model = VisionEncoderDecoderModel.from_pretrained(
            self.config.model_name, 
            revision="refs/pr/7"
        )
        
        # CẬP NHẬT DANH SÁCH TOKEN MỚI DỰA TRÊN DATA CỦA BẠN
        special_tokens = [
            "<s_resume>", "</s_resume>",
            "<s_skill>", "</s_skill>",
            "<s_person>", "</s_person>",
            "<s_education>", "</s_education>",
            "<s_designation>", "</s_designation>",
            "<s_company>", "</s_company>",
            "<s_email>", "</s_email>",
            "<s_location>", "</s_location>",
            "<s_projects>", "</s_projects>",
            "<s_certifications>", "</s_certifications>",
            "<s_languages>", "</s_languages>",
            "<s_awards>", "</s_awards>",
            "<s_volunteer>", "</s_volunteer>"
            
        ]
        processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        
        model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids("<s_resume>")
        model.config.pad_token_id = processor.tokenizer.pad_token_id
        model.decoder.config.max_position_embeddings = self.config.max_seq_length
        model.decoder.resize_token_embeddings(len(processor.tokenizer))
        
        return model.to(self.device), processor
    
    def prepare_datasets(self, processor: DonutProcessor) -> Tuple[Dataset, Dataset]:
        """Chuẩn bị dữ liệu Train/Val trực tiếp từ bộ nhớ đệm đĩa (Zero-RAM bloat)"""
        logger.info("Preparing datasets with RAM optimization...")
        
        if not self.config.use_synthetic_dataset:
            raise ValueError("Vui lòng kích hoạt use_synthetic_dataset để dùng dữ liệu local.")
            
        self.ensure_synthetic_dataset()

        # 1. Tải cấu trúc ImageFolder (Hugging Face quản lý bằng cơ chế ánh xạ bộ nhớ trên ổ đĩa, tốn ~0 MB RAM)
        logger.info(f"Loading imagefolder from {self.config.local_data_path}")
        hf_dataset = load_dataset("imagefolder", data_dir=self.config.local_data_path)
        
        # 2. Phân chia Train / Validation không dùng List
        if "validation" in hf_dataset and len(hf_dataset["validation"]) > 0:
            train_data = hf_dataset["train"]
            val_data = hf_dataset["validation"]
        else:
            logger.info(f"Tự động phân tách dữ liệu theo tỷ lệ validation: {self.config.validation_split}")
            split_dataset = hf_dataset["train"].train_test_split(
                test_size=self.config.validation_split, 
                seed=self.config.seed
            )
            train_data = split_dataset["train"]
            val_data = split_dataset["test"]
            
        logger.info(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")
        
        # 3. Bọc dữ liệu vào bộ nạp tối ưu RAM
        train_dataset = ResumeDocumentDataset(train_data, processor, self.config.max_seq_length)
        val_dataset = ResumeDocumentDataset(val_data, processor, self.config.max_seq_length)
        
        return train_dataset, val_dataset
    
    def train(self):
        """Fine-tune the Donut model."""
        model, processor = self.load_model_and_processor()
        train_dataset, eval_dataset = self.prepare_datasets(processor)
        
        # Thiết lập cấu hình huấn luyện
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            save_total_limit=3,
            eval_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            logging_steps=50,
            logging_dir="./logs",
            report_to=["none"],
            remove_unused_columns=False,  # Bắt buộc False để Seq2SeqTrainer nhận diện trường 'pixel_values'
            push_to_hub=False,
            seed=self.config.seed,
            bf16=True,                       # BẬT bf16 thay vì fp16 (Bắt buộc cho A100)
            fp16=False,                      # Tắt fp16 đi
            tf32=True,                       # Bật chế độ TF32 tính toán ma trận nhanh hơn trên Ampere
            dataloader_num_workers=8,        # Đa luồng hóa CPU xử lý ảnh (Lazy loading không lo nghẽn nữa)
            dataloader_pin_memory=True,      # Đẩy nhanh tốc độ truyền dữ liệu từ RAM lên VRAM của A100
        )
        
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.early_stopping_patience,
                    early_stopping_threshold=0.0,
                )
            ],
        )
        
        logger.info("Starting Donut fine-tuning...")
        trainer.train()
        
        logger.info(f"Saving final model to {self.config.output_dir}")
        model.save_pretrained(self.config.output_dir)
        processor.save_pretrained(self.config.output_dir)
        
        config_path = Path(self.config.output_dir) / "finetune_config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.__dict__, f, indent=2, default=str)
        
        logger.info("Donut fine-tuning completed successfully!")
        return model, processor


def main():
    """Main script entry point."""
    # Khởi tạo config tối ưu
    config = DonutConfig(
        num_epochs=50,
        batch_size=2,                
        gradient_accumulation_steps=8, 
        learning_rate=2e-5,
        use_synthetic_dataset=True,  
        use_kaggle_dataset=False,
        use_huggingface_dataset=False,
        local_data_path="./data/donut_dataset" 
    )
    
    fine_tuner = DonutFineTuner(config)
    try:
        model, processor = fine_tuner.train()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()
