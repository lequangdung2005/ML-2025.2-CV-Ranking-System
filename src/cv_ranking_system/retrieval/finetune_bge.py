"""
Fine-tune BGE-M3 (BAAI General Embedding Model) for CV/Resume ranking and retrieval.

This script fine-tunes the BGE-M3 model on resume-JD matching datasets to improve
semantic embedding and ranking capabilities.

Datasets:
- Resume-JD Match: https://huggingface.co/datasets/facehuggerapoorv/resume-jd-match
- Resume Score Details: https://huggingface.co/datasets/netsol/resume-score-details
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
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.models import Transformer, Pooling
from sentence_transformers import models

from cv_ranking_system.utils.logging_utils import setup_logger

logger = setup_logger(__name__)


@dataclass
class BGEConfig:
    """Configuration for BGE-M3 fine-tuning."""
    model_name: str = "BAAI/bge-m3"
    output_dir: str = "./models/bge-m3-finetuned"
    num_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    max_seq_length: int = 512
    weight_decay: float = 0.01
    validation_split: float = 0.1
    seed: int = 42
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    
    # Data sources
    use_resume_jd_dataset: bool = True
    use_resume_score_dataset: bool = True
    local_data_path: Optional[str] = None
    
    # Training options
    loss_type: str = "contrastive"  # contrastive or triplet
    gradient_accumulation_steps: int = 8
    save_steps: int = 500
    eval_steps: int = 250
    early_stopping_patience: int = 3
    
    # Embedding options
    use_sentence_transformers: bool = True  # Use Sentence Transformers for easier training


class ResumeJDDataset(Dataset):
    """Dataset for resume-JD matching pairs."""
    
    def __init__(
        self,
        resumes: List[str],
        job_descriptions: List[str],
        scores: List[float],
        tokenizer,
        max_seq_length: int = 512,
    ):
        """
        Initialize the dataset.
        
        Args:
            resumes: List of resume texts
            job_descriptions: List of job description texts
            scores: Matching scores (0-1)
            tokenizer: Tokenizer for encoding
            max_seq_length: Maximum token sequence length
        """
        self.resumes = resumes
        self.job_descriptions = job_descriptions
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        
        assert len(resumes) == len(job_descriptions) == len(scores)
    
    def __len__(self) -> int:
        return len(self.resumes)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a sample from the dataset."""
        resume = self.resumes[idx]
        jd = self.job_descriptions[idx]
        score = self.scores[idx]
        
        # Create positive/negative examples based on score threshold
        is_positive = 1 if score > 0.5 else 0
        
        # Tokenize
        resume_tokens = self.tokenizer(
            resume,
            max_length=self.max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        jd_tokens = self.tokenizer(
            jd,
            max_length=self.max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "resume_ids": resume_tokens["input_ids"].squeeze(0),
            "resume_mask": resume_tokens["attention_mask"].squeeze(0),
            "jd_ids": jd_tokens["input_ids"].squeeze(0),
            "jd_mask": jd_tokens["attention_mask"].squeeze(0),
            "score": torch.tensor(score, dtype=torch.float),
            "label": torch.tensor(is_positive, dtype=torch.long),
        }


class ContrastiveDataset(Dataset):
    """Dataset for contrastive learning with triplets."""
    
    def __init__(
        self,
        resumes: List[str],
        job_descriptions: List[str],
        scores: List[float],
        tokenizer,
        max_seq_length: int = 512,
    ):
        """Initialize contrastive dataset with positive and negative samples."""
        self.data = []
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        
        # Group by resume
        resume_to_jds = {}
        for resume, jd, score in zip(resumes, job_descriptions, scores):
            if resume not in resume_to_jds:
                resume_to_jds[resume] = {"positive": [], "negative": []}
            
            if score > 0.7:
                resume_to_jds[resume]["positive"].append(jd)
            elif score < 0.3:
                resume_to_jds[resume]["negative"].append(jd)
        
        # Create triplets
        for resume, jds in resume_to_jds.items():
            if jds["positive"] and jds["negative"]:
                for pos_jd in jds["positive"]:
                    for neg_jd in jds["negative"]:
                        self.data.append((resume, pos_jd, neg_jd))
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a triplet from the dataset."""
        anchor, positive, negative = self.data[idx]
        
        anchor_tokens = self.tokenizer(
            anchor,
            max_length=self.max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        pos_tokens = self.tokenizer(
            positive,
            max_length=self.max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        neg_tokens = self.tokenizer(
            negative,
            max_length=self.max_seq_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "anchor_ids": anchor_tokens["input_ids"].squeeze(0),
            "anchor_mask": anchor_tokens["attention_mask"].squeeze(0),
            "positive_ids": pos_tokens["input_ids"].squeeze(0),
            "positive_mask": pos_tokens["attention_mask"].squeeze(0),
            "negative_ids": neg_tokens["input_ids"].squeeze(0),
            "negative_mask": neg_tokens["attention_mask"].squeeze(0),
        }


class BGEFineTuner:
    """Handler for BGE-M3 model fine-tuning."""
    
    def __init__(self, config: BGEConfig):
        """Initialize the fine-tuner."""
        self.config = config
        self.device = torch.device(config.device)
        
        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Set random seed
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        
        logger.info(f"Initializing BGE-M3 fine-tuner on {self.device}")
        logger.info(f"Config: {config}")
    
    def load_resume_jd_dataset(self) -> Tuple[List[str], List[str], List[float]]:
        """Load resume-JD matching dataset."""
        logger.info("Loading resume-JD match dataset...")
        try:
            dataset = load_dataset("facehuggerapoorv/resume-jd-match")
            
            resumes = []
            jds = []
            scores = []
            
            for sample in tqdm(dataset["train"], desc="Processing resume-JD dataset"):
                if "resume" in sample and "job_description" in sample:
                    resumes.append(sample["resume"])
                    jds.append(sample["job_description"])
                    # Normalize score to 0-1
                    score = sample.get("match_score", 0.5)
                    if isinstance(score, str):
                        score = float(score)
                    scores.append(min(max(score / 100, 0), 1))  # Clamp to [0, 1]
            
            logger.info(f"Loaded {len(resumes)} resume-JD pairs")
            return resumes, jds, scores
        
        except Exception as e:
            logger.warning(f"Failed to load resume-JD dataset: {e}")
            return [], [], []
    
    def load_resume_score_dataset(self) -> Tuple[List[str], List[str], List[float]]:
        """Load resume scoring dataset."""
        logger.info("Loading resume score details dataset...")
        try:
            dataset = load_dataset("netsol/resume-score-details")
            
            resumes = []
            jds = []
            scores = []
            
            for sample in tqdm(dataset["train"], desc="Processing resume score dataset"):
                if "resume_text" in sample and "job_description" in sample:
                    resumes.append(sample["resume_text"])
                    jds.append(sample["job_description"])
                    
                    # Extract score (normalize if needed)
                    score = sample.get("match_score", sample.get("score", 0.5))
                    if isinstance(score, str):
                        score = float(score)
                    scores.append(min(max(score, 0), 1))
            
            logger.info(f"Loaded {len(resumes)} resume-score pairs")
            return resumes, jds, scores
        
        except Exception as e:
            logger.warning(f"Failed to load resume score dataset: {e}")
            return [], [], []
    
    def load_local_dataset(self, data_path: str) -> Tuple[List[str], List[str], List[float]]:
        """Load dataset from local JSONL file."""
        logger.info(f"Loading local dataset from {data_path}")
        
        resumes = []
        jds = []
        scores = []
        
        data_file = Path(data_path)
        if not data_file.exists():
            logger.warning(f"Data file not found: {data_path}")
            return [], [], []
        
        with open(data_file, "r") as f:
            for line in tqdm(f, desc="Loading local data"):
                try:
                    item = json.loads(line)
                    resumes.append(item["resume"])
                    jds.append(item["job_description"])
                    scores.append(float(item["score"]))
                except (json.JSONDecodeError, KeyError) as e:
                    logger.debug(f"Skipped invalid line: {e}")
        
        logger.info(f"Loaded {len(resumes)} local samples")
        return resumes, jds, scores
    
    def prepare_datasets(self) -> Tuple[Dataset, Dataset]:
        """Prepare training and validation datasets."""
        logger.info("Preparing datasets...")
        
        all_resumes = []
        all_jds = []
        all_scores = []
        
        # Load from multiple sources
        if self.config.use_resume_jd_dataset:
            resumes, jds, scores = self.load_resume_jd_dataset()
            all_resumes.extend(resumes)
            all_jds.extend(jds)
            all_scores.extend(scores)
        
        if self.config.use_resume_score_dataset:
            resumes, jds, scores = self.load_resume_score_dataset()
            all_resumes.extend(resumes)
            all_jds.extend(jds)
            all_scores.extend(scores)
        
        if self.config.local_data_path:
            resumes, jds, scores = self.load_local_dataset(self.config.local_data_path)
            all_resumes.extend(resumes)
            all_jds.extend(jds)
            all_scores.extend(scores)
        
        if not all_resumes:
            raise ValueError("No datasets loaded. Please configure at least one data source.")
        
        logger.info(f"Total samples: {len(all_resumes)}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        # Split into train/val
        num_train = int(len(all_resumes) * (1 - self.config.validation_split))
        indices = np.random.permutation(len(all_resumes))
        
        train_indices = indices[:num_train]
        val_indices = indices[num_train:]
        
        # Choose dataset class based on loss type
        if self.config.loss_type == "triplet":
            train_dataset = ContrastiveDataset(
                [all_resumes[i] for i in train_indices],
                [all_jds[i] for i in train_indices],
                [all_scores[i] for i in train_indices],
                tokenizer,
                self.config.max_seq_length,
            )
            val_dataset = ContrastiveDataset(
                [all_resumes[i] for i in val_indices],
                [all_jds[i] for i in val_indices],
                [all_scores[i] for i in val_indices],
                tokenizer,
                self.config.max_seq_length,
            )
        else:
            train_dataset = ResumeJDDataset(
                [all_resumes[i] for i in train_indices],
                [all_jds[i] for i in train_indices],
                [all_scores[i] for i in train_indices],
                tokenizer,
                self.config.max_seq_length,
            )
            val_dataset = ResumeJDDataset(
                [all_resumes[i] for i in val_indices],
                [all_jds[i] for i in val_indices],
                [all_scores[i] for i in val_indices],
                tokenizer,
                self.config.max_seq_length,
            )
        
        logger.info(f"Train set: {len(train_dataset)}, Val set: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def train_with_sentence_transformers(self):
        """Train using Sentence Transformers library (easier API)."""
        logger.info("Training with Sentence Transformers...")
        
        # Load all data for training pairs
        resumes = []
        jds = []
        scores = []
        
        if self.config.use_resume_jd_dataset:
            r, j, s = self.load_resume_jd_dataset()
            resumes.extend(r)
            jds.extend(j)
            scores.extend(s)
        
        if self.config.use_resume_score_dataset:
            r, j, s = self.load_resume_score_dataset()
            resumes.extend(r)
            jds.extend(j)
            scores.extend(s)
        
        if self.config.local_data_path:
            r, j, s = self.load_local_dataset(self.config.local_data_path)
            resumes.extend(r)
            jds.extend(j)
            scores.extend(s)
        
        # Load model
        model = SentenceTransformer(self.config.model_name)
        
        # Prepare training examples
        train_examples = []
        for resume, jd, score in zip(resumes, jds, scores):
            # Positive pairs (high score) and negative pairs (low score)
            train_examples.append(InputExample(
                texts=[resume, jd],
                label=score
            ))
        
        # Split train/val
        num_train = int(len(train_examples) * (1 - self.config.validation_split))
        train_examples = train_examples[:num_train]
        val_examples = train_examples[num_train:]
        
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=self.config.batch_size)
        
        # Define loss
        if self.config.loss_type == "triplet":
            train_loss = losses.TripletLoss(model)
        else:
            train_loss = losses.CosineSimilarityLoss(model)
        
        # Train
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=self.config.num_epochs,
            warmup_steps=int(len(train_dataloader) * self.config.num_epochs * self.config.warmup_ratio),
            evaluation_steps=self.config.eval_steps,
            evaluator=None,
            output_path=self.config.output_dir,
            save_best_model=True,
            checkpoint_save_steps=self.config.save_steps,
            checkpoint_save_total_limit=3,
        )
        
        logger.info("Training completed!")
        return model
    
    def train_with_transformers(self):
        """Train using transformers library (lower level)."""
        logger.info("Training with Transformers library...")
        
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        model = AutoModel.from_pretrained(self.config.model_name).to(self.device)
        
        # Prepare datasets
        train_dataset, val_dataset = self.prepare_datasets()
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
        )
        
        scaler = torch.amp.GradScaler("cuda")
        
        optimizer = AdamW(model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        
        # Tính toán lại tổng số step dựa trên gradient_accumulation_steps
        num_training_steps = (len(train_dataloader) // self.config.gradient_accumulation_steps) * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(num_training_steps * self.config.warmup_ratio),
            num_training_steps=num_training_steps,
        )
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            model.train()
            total_loss = 0
            optimizer.zero_grad() # Đưa ra ngoài vòng lặp batch để tích lũy
            
            for step, batch in enumerate(tqdm(train_dataloader, desc="Training")):
                # Chạy forward pass với autocast (FP16)
                with torch.amp.autocast("cuda"):
                    resume_embeddings = self._encode_batch(model, batch, "resume", tokenizer)
                    jd_embeddings = self._encode_batch(model, batch, "jd", tokenizer)
                    
                    similarity = torch.nn.functional.cosine_similarity(resume_embeddings, jd_embeddings)
                    scores = batch["score"].to(self.device)
                    
                    # Chia loss cho accumulation steps
                    loss = torch.nn.functional.mse_loss(similarity, scores)
                    loss = loss / self.config.gradient_accumulation_steps
                
                # Backward pass có scale để tránh underflow gradient ở FP16
                scaler.scale(loss).backward()
                total_loss += loss.item() * self.config.gradient_accumulation_steps
                
                # Thực hiện cập nhật trọng số sau khi tích lũy đủ bước
                if (step + 1) % self.config.gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
            
            avg_loss = total_loss / len(train_dataloader)
            logger.info(f"Average training loss: {avg_loss:.4f}")
                        
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc="Validating"):
                    resume_embeddings = self._encode_batch(model, batch, "resume", tokenizer)
                    jd_embeddings = self._encode_batch(model, batch, "jd", tokenizer)
                    
                    similarity = torch.nn.functional.cosine_similarity(resume_embeddings, jd_embeddings)
                    scores = batch["score"].to(self.device)
                    
                    loss = torch.nn.functional.mse_loss(similarity, scores)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(val_dataloader)
            logger.info(f"Average validation loss: {avg_val_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % 1 == 0:
                checkpoint_dir = Path(self.config.output_dir) / f"checkpoint-{epoch + 1}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
        
        # Save final model
        logger.info(f"Saving model to {self.config.output_dir}")
        model.save_pretrained(self.config.output_dir)
        tokenizer.save_pretrained(self.config.output_dir)
        
        return model
    
    def _encode_batch(self, model, batch, prefix, tokenizer):
        """Encode a batch of text."""
        input_ids = batch[f"{prefix}_ids"].to(self.device)
        attention_mask = batch[f"{prefix}_mask"].to(self.device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state[:, 0]  # [CLS] token
        
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    def train(self):
        """Fine-tune the BGE-M3 model."""
        if self.config.use_sentence_transformers:
            return self.train_with_sentence_transformers()
        else:
            return self.train_with_transformers()


def main():
    """Main fine-tuning script."""
    # Configure
    config = BGEConfig(
        num_epochs=5,
        batch_size=4,
        learning_rate=2e-5,
        loss_type="contrastive",
        use_sentence_transformers=True,
        use_resume_jd_dataset=True,
        use_resume_score_dataset=True,
    )
    
    # Fine-tune
    fine_tuner = BGEFineTuner(config)
    model = fine_tuner.train()
    
    logger.info("BGE-M3 fine-tuning pipeline completed!")


if __name__ == "__main__":
    main()
