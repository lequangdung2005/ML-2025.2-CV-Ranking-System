"""
Fine-tune Donut (Document Understanding Transformer) for CV/Resume extraction.

This script fine-tunes the Donut model on resume NER datasets to improve
OCR and information extraction capabilities.

Datasets:
- Resume NER Training Dataset: https://www.kaggle.com/datasets/yashpwrr/resume-ner-training-dataset
- Resume Corpus Dataset: https://github.com/vrundag91/Resume-Corpus-Dataset
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    VisionEncoderDecoderModel,
    DonutProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
)
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from cv_ranking_system.utils.logging_utils import setup_logger

logger = setup_logger(__name__)


@dataclass
class DonutConfig:
    """Configuration for Donut fine-tuning."""
    model_name: str = "naver-clova-ix/donut-base"
    output_dir: str = "./models/donut-finetuned"
    num_epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-4
    max_seq_length: int = 768
    image_size: Tuple[int, int] = field(default_factory=lambda: (1280, 960))
    warmup_steps: int = 500
    weight_decay: float = 0.01
    validation_split: float = 0.1
    seed: int = 42
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    
    # Data sources
    use_kaggle_dataset: bool = True
    use_huggingface_dataset: bool = True
    local_data_path: Optional[str] = None
    
    # Training options
    gradient_accumulation_steps: int = 2
    max_steps: int = -1
    save_steps: int = 500
    eval_steps: int = 250
    early_stopping_patience: int = 3


class ResumeDocumentDataset(Dataset):
    """Dataset for resume documents with OCR/extraction annotations."""
    
    def __init__(
        self,
        images: List[str],
        texts: List[str],
        processor: DonutProcessor,
        max_seq_length: int = 768,
        image_size: Tuple[int, int] = (1280, 960),
    ):
        """
        Initialize the dataset.
        
        Args:
            images: List of image paths or PIL Images
            texts: List of text/JSON annotations
            processor: DonutProcessor for preprocessing
            max_seq_length: Maximum token sequence length
            image_size: Target image size
        """
        self.images = images
        self.texts = texts
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.image_size = image_size
        
        assert len(images) == len(texts), "Number of images and texts must match"
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a sample from the dataset."""
        # Load image
        if isinstance(self.images[idx], str):
            image = Image.open(self.images[idx]).convert("RGB")
        else:
            image = self.images[idx]
        
        # Get text/annotation
        text = self.texts[idx]
        
        # Process image
        pixel_values = self.processor(
            image, 
            return_tensors="pt"
        ).pixel_values.squeeze(0)
        
        # Process text - format as instruction for Donut
        # Donut uses <s_resume> for resume understanding
        full_text = f"<s_resume>{text}</s_resume>"
        
        # Tokenize with truncation
        input_ids = self.processor.tokenizer(
            full_text,
            max_length=self.max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids["input_ids"].squeeze(0),
            "attention_mask": input_ids.get("attention_mask", torch.ones_like(input_ids["input_ids"])).squeeze(0),
            "decoder_input_ids": input_ids["input_ids"].squeeze(0),
            "decoder_attention_mask": input_ids.get("attention_mask", torch.ones_like(input_ids["input_ids"])).squeeze(0),
        }


class DonutFineTuner:
    """Handler for Donut model fine-tuning."""
    
    def __init__(self, config: DonutConfig):
        """Initialize the fine-tuner."""
        self.config = config
        self.device = torch.device(config.device)
        
        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Set random seed
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        logger.info(f"Initializing Donut fine-tuner on {self.device}")
        logger.info(f"Config: {config}")
    
    def load_model_and_processor(self) -> Tuple[VisionEncoderDecoderModel, DonutProcessor]:
        """Load pre-trained Donut model and processor."""
        logger.info(f"Loading model: {self.config.model_name}")
        
        processor = DonutProcessor.from_pretrained(self.config.model_name)
        model = VisionEncoderDecoderModel.from_pretrained(self.config.model_name)
        
        # Update decoder config for resume extraction task
        model.decoder.config.max_position_embeddings = self.config.max_seq_length
        model.decoder.config.vocab_size = len(processor.tokenizer)
        
        return model.to(self.device), processor
    
    def load_kaggle_dataset(self) -> Tuple[List[str], List[str]]:
        """Load resume NER dataset from Kaggle."""
        logger.info("Loading Kaggle resume NER dataset...")
        try:
            dataset = load_dataset("yashpwrr/resume-ner-training-dataset")
            
            images = []
            texts = []
            
            # Process dataset - structure depends on actual Kaggle dataset format
            for sample in tqdm(dataset["train"], desc="Processing Kaggle dataset"):
                # Assuming the dataset has 'image' and 'text' or 'ner_tags' fields
                if "image" in sample and "text" in sample:
                    images.append(sample["image"])
                    texts.append(json.dumps({"text": sample["text"], "ner_tags": sample.get("ner_tags", [])}))
            
            logger.info(f"Loaded {len(images)} samples from Kaggle dataset")
            return images, texts
        
        except Exception as e:
            logger.warning(f"Failed to load Kaggle dataset: {e}")
            return [], []
    
    def load_huggingface_dataset(self) -> Tuple[List[str], List[str]]:
        """Load resume corpus dataset from HuggingFace."""
        logger.info("Loading HuggingFace resume dataset...")
        try:
            # Note: The exact dataset identifier may need adjustment based on availability
            dataset = load_dataset("datasetmaster/resumes", split="train")
            
            images = []
            texts = []
            
            for sample in tqdm(dataset, desc="Processing HuggingFace dataset"):
                if "image" in sample and "text" in sample:
                    images.append(sample["image"])
                    texts.append(json.dumps({
                        "text": sample["text"],
                        "metadata": sample.get("metadata", {})
                    }))
            
            logger.info(f"Loaded {len(images)} samples from HuggingFace dataset")
            return images, texts
        
        except Exception as e:
            logger.warning(f"Failed to load HuggingFace dataset: {e}")
            return [], []
    
    def load_local_dataset(self, data_path: str) -> Tuple[List[str], List[str]]:
        """Load dataset from local directory."""
        logger.info(f"Loading local dataset from {data_path}")
        
        images = []
        texts = []
        
        data_dir = Path(data_path)
        
        # Expect structure: data_dir/images/ and data_dir/annotations.jsonl
        images_dir = data_dir / "images"
        annotations_file = data_dir / "annotations.jsonl"
        
        if not images_dir.exists() or not annotations_file.exists():
            logger.warning(f"Expected directory structure not found in {data_path}")
            return [], []
        
        with open(annotations_file, "r") as f:
            for line in tqdm(f, desc="Loading local annotations"):
                annotation = json.loads(line)
                image_path = images_dir / annotation["image_name"]
                
                if image_path.exists():
                    images.append(str(image_path))
                    texts.append(json.dumps(annotation.get("extracted_text", "")))
        
        logger.info(f"Loaded {len(images)} samples from local dataset")
        return images, texts
    
    def prepare_datasets(self) -> Tuple[Dataset, Dataset]:
        """Prepare training and validation datasets."""
        logger.info("Preparing datasets...")
        
        all_images = []
        all_texts = []
        
        # Load from multiple sources
        if self.config.use_kaggle_dataset:
            images, texts = self.load_kaggle_dataset()
            all_images.extend(images)
            all_texts.extend(texts)
        
        if self.config.use_huggingface_dataset:
            images, texts = self.load_huggingface_dataset()
            all_images.extend(images)
            all_texts.extend(texts)
        
        if self.config.local_data_path:
            images, texts = self.load_local_dataset(self.config.local_data_path)
            all_images.extend(images)
            all_texts.extend(texts)
        
        if not all_images:
            raise ValueError("No datasets loaded. Please configure at least one data source.")
        
        logger.info(f"Total samples: {len(all_images)}")
        
        # Load processor
        processor = DonutProcessor.from_pretrained(self.config.model_name)
        
        # Split into train/val
        num_train = int(len(all_images) * (1 - self.config.validation_split))
        indices = np.random.permutation(len(all_images))
        
        train_indices = indices[:num_train]
        val_indices = indices[num_train:]
        
        train_dataset = ResumeDocumentDataset(
            [all_images[i] for i in train_indices],
            [all_texts[i] for i in train_indices],
            processor,
            self.config.max_seq_length,
            self.config.image_size,
        )
        
        val_dataset = ResumeDocumentDataset(
            [all_images[i] for i in val_indices],
            [all_texts[i] for i in val_indices],
            processor,
            self.config.max_seq_length,
            self.config.image_size,
        )
        
        logger.info(f"Train set: {len(train_dataset)}, Val set: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def train(self):
        """Fine-tune the Donut model."""
        # Load model and processor
        model, processor = self.load_model_and_processor()
        
        # Prepare datasets
        train_dataset, eval_dataset = self.prepare_datasets()
        
        # Training arguments
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
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            logging_steps=100,
            logging_dir="./logs",
            report_to=["tensorboard"],
            remove_unused_columns=False,
            push_to_hub=False,
            seed=self.config.seed,
            fp16=torch.cuda.is_available(),
        )
        
        # Initialize trainer
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
        
        # Train
        logger.info("Starting fine-tuning...")
        trainer.train()
        
        # Save final model
        logger.info(f"Saving model to {self.config.output_dir}")
        model.save_pretrained(self.config.output_dir)
        processor.save_pretrained(self.config.output_dir)
        
        # Save config
        config_path = Path(self.config.output_dir) / "finetune_config.json"
        with open(config_path, "w") as f:
            json.dump(self.config.__dict__, f, indent=2, default=str)
        
        logger.info("Fine-tuning completed successfully!")
        
        return model, processor


def main():
    """Main fine-tuning script."""
    # Configure
    config = DonutConfig(
        num_epochs=10,
        batch_size=8,
        learning_rate=1e-4,
        use_kaggle_dataset=True,
        use_huggingface_dataset=True,
    )
    
    # Fine-tune
    fine_tuner = DonutFineTuner(config)
    model, processor = fine_tuner.train()
    
    logger.info("Donut fine-tuning pipeline completed!")


if __name__ == "__main__":
    main()
