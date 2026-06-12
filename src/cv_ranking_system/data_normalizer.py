"""
Data Normalization and Preprocessing Module for Fine-Tuning Pipeline

Handles comprehensive data normalization including:
- Text cleaning and standardization
- Image normalization and augmentation
- Statistical normalization
- Data validation and quality checks
- Outlier detection and handling
- Data balancing
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageEnhance
from tqdm import tqdm

from cv_ranking_system.utils.logging_utils import setup_logger

logger = setup_logger(__name__)


@dataclass
class NormalizationConfig:
    """Configuration for data normalization."""
    # Text normalization
    lowercase: bool = True
    remove_extra_whitespace: bool = True
    remove_special_chars: bool = False
    remove_numbers: bool = False
    remove_punctuation: bool = False
    normalize_unicode: bool = True
    min_text_length: int = 10
    max_text_length: int = 10000
    
    # Image normalization
    target_image_size: Tuple[int, int] = (1280, 960)
    image_format: str = "RGB"
    normalize_pixel_values: bool = True
    apply_augmentation: bool = False
    
    # Statistical normalization
    normalize_scores: bool = True
    score_min: float = 0.0
    score_max: float = 1.0
    
    # Data validation
    remove_duplicates: bool = True
    handle_missing_values: bool = True
    detect_outliers: bool = True
    outlier_threshold: float = 3.0  # Standard deviations
    
    # Data balancing
    balance_data: bool = False
    balance_strategy: str = "oversample"  # oversample or undersample


class TextNormalizer:
    """Handle text normalization and cleaning."""
    
    def __init__(self, config: NormalizationConfig):
        """Initialize text normalizer."""
        self.config = config
        
        # Common abbreviations mapping
        self.abbreviations = {
            r'\bDr\.': 'Doctor',
            r'\bMr\.': 'Mister',
            r'\bMs\.': 'Miss',
            r'\bProf\.': 'Professor',
            r'\bCEO\b': 'Chief Executive Officer',
            r'\bCTO\b': 'Chief Technology Officer',
            r'\bQA\b': 'Quality Assurance',
        }
    
    def normalize(self, text: str) -> str:
        """Normalize a single text."""
        if not isinstance(text, str):
            return ""
        
        # Remove extra whitespace
        if self.config.remove_extra_whitespace:
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
        
        # Normalize unicode
        if self.config.normalize_unicode:
            text = self._normalize_unicode(text)
        
        # Remove special characters (except essential ones)
        if self.config.remove_special_chars:
            text = re.sub(r'[^a-zA-Z0-9\s\-\.,@]', '', text)
        
        # Remove numbers
        if self.config.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Remove punctuation
        if self.config.remove_punctuation:
            text = re.sub(r'[^\w\s]', '', text)
        
        # Lowercase
        if self.config.lowercase:
            text = text.lower()
        
        # Expand abbreviations
        for abbrev, full in self.abbreviations.items():
            text = re.sub(abbrev, full, text, flags=re.IGNORECASE)
        
        # Remove extra whitespace again after cleaning
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters."""
        import unicodedata
        # NFKD = Compatibility Decomposition
        return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    
    def validate_text(self, text: str) -> bool:
        """Validate text meets requirements."""
        if not text:
            return False
        
        length = len(text)
        
        if length < self.config.min_text_length:
            logger.debug(f"Text too short ({length} < {self.config.min_text_length})")
            return False
        
        if length > self.config.max_text_length:
            logger.debug(f"Text too long ({length} > {self.config.max_text_length})")
            return False
        
        return True
    
    def normalize_batch(self, texts: List[str]) -> Tuple[List[str], List[int]]:
        """Normalize batch of texts and return valid indices."""
        normalized = []
        valid_indices = []
        
        for idx, text in enumerate(tqdm(texts, desc="Normalizing texts")):
            norm_text = self.normalize(text)
            
            if self.validate_text(norm_text):
                normalized.append(norm_text)
                valid_indices.append(idx)
            else:
                logger.debug(f"Skipped invalid text at index {idx}")
        
        logger.info(f"Normalized {len(normalized)}/{len(texts)} texts")
        
        return normalized, valid_indices


class ImageNormalizer:
    """Handle image normalization and preprocessing."""
    
    def __init__(self, config: NormalizationConfig):
        """Initialize image normalizer."""
        self.config = config
    
    def normalize(self, image: Union[str, Image.Image]) -> Image.Image:
        """Normalize a single image."""
        # Load image if path
        if isinstance(image, str):
            try:
                image = Image.open(image)
            except Exception as e:
                logger.error(f"Failed to load image {image}: {e}")
                return None
        
        # Convert to target format
        image = image.convert(self.config.image_format)
        
        # Resize image
        image = ImageOps.fit(
            image,
            self.config.target_image_size,
            Image.Resampling.LANCZOS,
            centering=(0.5, 0.5)
        )
        
        return image
    
    def validate_image(self, image: Image.Image) -> bool:
        """Validate image meets requirements."""
        if image is None:
            return False
        
        # Check image size
        if image.size != self.config.target_image_size:
            logger.debug(f"Invalid image size: {image.size}")
            return False
        
        # Check image mode
        if image.mode != self.config.image_format:
            logger.debug(f"Invalid image format: {image.mode}")
            return False
        
        # Check for corruption (basic check)
        try:
            image.tobytes()
        except Exception as e:
            logger.debug(f"Corrupted image: {e}")
            return False
        
        return True
    
    def apply_augmentation(self, image: Image.Image) -> Image.Image:
        """Apply light augmentation for regularization."""
        if not self.config.apply_augmentation:
            return image
        
        # Random rotation (small)
        if np.random.rand() > 0.7:
            angle = np.random.uniform(-5, 5)
            image = image.rotate(angle, fillcolor='white')
        
        # Random brightness
        if np.random.rand() > 0.7:
            enhancer = ImageEnhance.Brightness(image)
            factor = np.random.uniform(0.9, 1.1)
            image = enhancer.enhance(factor)
        
        # Random contrast
        if np.random.rand() > 0.7:
            enhancer = ImageEnhance.Contrast(image)
            factor = np.random.uniform(0.9, 1.1)
            image = enhancer.enhance(factor)
        
        return image
    
    def normalize_batch(self, images: List[Union[str, Image.Image]]) -> Tuple[List[Image.Image], List[int]]:
        """Normalize batch of images and return valid indices."""
        normalized = []
        valid_indices = []
        
        for idx, image in enumerate(tqdm(images, desc="Normalizing images")):
            try:
                norm_image = self.normalize(image)
                
                if self.validate_image(norm_image):
                    if self.config.apply_augmentation:
                        norm_image = self.apply_augmentation(norm_image)
                    
                    normalized.append(norm_image)
                    valid_indices.append(idx)
            except Exception as e:
                logger.debug(f"Failed to normalize image at index {idx}: {e}")
        
        logger.info(f"Normalized {len(normalized)}/{len(images)} images")
        
        return normalized, valid_indices


class ScoreNormalizer:
    """Handle score/label normalization."""
    
    def __init__(self, config: NormalizationConfig):
        """Initialize score normalizer."""
        self.config = config
    
    def normalize(self, scores: List[float]) -> List[float]:
        """Normalize scores to target range."""
        if not self.config.normalize_scores:
            return scores
        
        scores = np.array(scores)
        
        # Handle missing values
        if self.config.handle_missing_values:
            scores = np.nan_to_num(scores, nan=np.nanmean(scores))
        
        # Min-Max normalization
        min_val = np.min(scores)
        max_val = np.max(scores)
        
        if max_val > min_val:
            normalized = (scores - min_val) / (max_val - min_val)
        else:
            normalized = np.full_like(scores, 0.5)
        
        # Scale to target range
        normalized = normalized * (self.config.score_max - self.config.score_min) + self.config.score_min
        
        return normalized.tolist()
    
    def validate_score(self, score: float) -> bool:
        """Validate score is in valid range."""
        return self.config.score_min <= score <= self.config.score_max


class DataValidator:
    """Validate and clean data."""
    
    def __init__(self, config: NormalizationConfig):
        """Initialize validator."""
        self.config = config
    
    def remove_duplicates(
        self,
        texts: List[str],
        indices: Optional[List[int]] = None,
    ) -> Tuple[List[str], List[int]]:
        """Remove duplicate texts."""
        if not self.config.remove_duplicates:
            return texts, indices or list(range(len(texts)))
        
        unique_texts = []
        unique_indices = []
        seen = set()
        
        for idx, text in enumerate(texts):
            text_hash = hash(text)
            
            if text_hash not in seen:
                seen.add(text_hash)
                unique_texts.append(text)
                
                if indices:
                    unique_indices.append(indices[idx])
                else:
                    unique_indices.append(idx)
        
        logger.info(f"Removed {len(texts) - len(unique_texts)} duplicates")
        
        return unique_texts, unique_indices
    
    def detect_outliers(
        self,
        values: List[float],
        indices: Optional[List[int]] = None,
    ) -> Tuple[List[float], List[int]]:
        """Detect and remove outliers using z-score."""
        if not self.config.detect_outliers:
            return values, indices or list(range(len(values)))
        
        values = np.array(values)
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return values.tolist(), indices or list(range(len(values)))
        
        z_scores = np.abs((values - mean) / std)
        valid_mask = z_scores <= self.config.outlier_threshold
        
        filtered_values = values[valid_mask].tolist()
        
        if indices:
            filtered_indices = [idx for idx, valid in zip(indices, valid_mask) if valid]
        else:
            filtered_indices = [idx for idx, valid in enumerate(valid_mask) if valid]
        
        logger.info(f"Removed {len(values) - len(filtered_values)} outliers")
        
        return filtered_values, filtered_indices
    
    def validate_dataset(
        self,
        texts: List[str],
        values: Optional[List[float]] = None,
        images: Optional[List[str]] = None,
    ) -> Dict:
        """Validate entire dataset."""
        report = {
            "total_samples": len(texts),
            "valid_texts": 0,
            "valid_values": 0,
            "valid_images": 0,
            "issues": [],
        }
        
        # Validate texts
        valid_texts = [t for t in texts if t and len(t) > 0]
        report["valid_texts"] = len(valid_texts)
        
        if len(valid_texts) < len(texts):
            report["issues"].append(f"Found {len(texts) - len(valid_texts)} invalid texts")
        
        # Validate values
        if values:
            valid_values = [v for v in values if not np.isnan(v) and not np.isinf(v)]
            report["valid_values"] = len(valid_values)
            
            if len(valid_values) < len(values):
                report["issues"].append(f"Found {len(values) - len(valid_values)} invalid values")
        
        # Validate images
        if images:
            valid_images = 0
            for img_path in images:
                try:
                    Image.open(img_path)
                    valid_images += 1
                except:
                    pass
            
            report["valid_images"] = valid_images
            
            if valid_images < len(images):
                report["issues"].append(f"Found {len(images) - valid_images} invalid images")
        
        return report


class DataBalancer:
    """Handle data balancing."""
    
    def __init__(self, config: NormalizationConfig):
        """Initialize balancer."""
        self.config = config
    
    def balance(
        self,
        data: List,
        labels: List[int],
    ) -> Tuple[List, List[int]]:
        """Balance dataset by class."""
        if not self.config.balance_data:
            return data, labels
        
        labels = np.array(labels)
        unique_labels = np.unique(labels)
        
        if len(unique_labels) == 1:
            logger.info("Dataset has only one class, no balancing needed")
            return data, labels.tolist()
        
        # Count samples per class
        class_counts = Counter(labels)
        logger.info(f"Class distribution before balancing: {dict(class_counts)}")
        
        if self.config.balance_strategy == "oversample":
            balanced_data, balanced_labels = self._oversample(data, labels, class_counts)
        else:
            balanced_data, balanced_labels = self._undersample(data, labels, class_counts)
        
        logger.info(f"Class distribution after balancing: {Counter(balanced_labels)}")
        
        return balanced_data, balanced_labels
    
    def _oversample(self, data, labels, class_counts):
        """Oversample minority classes."""
        max_count = max(class_counts.values())
        balanced_data = []
        balanced_labels = []
        
        for label in np.unique(labels):
            class_indices = np.where(labels == label)[0]
            num_samples = len(class_indices)
            
            # Keep original samples
            for idx in class_indices:
                balanced_data.append(data[idx])
                balanced_labels.append(label)
            
            # Oversample if needed
            num_to_add = max_count - num_samples
            if num_to_add > 0:
                additional_indices = np.random.choice(class_indices, size=num_to_add, replace=True)
                for idx in additional_indices:
                    balanced_data.append(data[idx])
                    balanced_labels.append(label)
        
        return balanced_data, np.array(balanced_labels).tolist()
    
    def _undersample(self, data, labels, class_counts):
        """Undersample majority classes."""
        min_count = min(class_counts.values())
        balanced_data = []
        balanced_labels = []
        
        for label in np.unique(labels):
            class_indices = np.where(labels == label)[0]
            
            # Undersample to minimum class size
            selected_indices = np.random.choice(class_indices, size=min_count, replace=False)
            
            for idx in selected_indices:
                balanced_data.append(data[idx])
                balanced_labels.append(label)
        
        return balanced_data, np.array(balanced_labels).tolist()


class DataNormalizer:
    """Main data normalization orchestrator."""
    
    def __init__(self, config: NormalizationConfig):
        """Initialize normalizer."""
        self.config = config
        self.text_normalizer = TextNormalizer(config)
        self.image_normalizer = ImageNormalizer(config)
        self.score_normalizer = ScoreNormalizer(config)
        self.validator = DataValidator(config)
        self.balancer = DataBalancer(config)
        
        logger.info(f"Initialized DataNormalizer with config: {config}")
    
    def normalize_extraction_data(
        self,
        images: List[Union[str, Image.Image]],
        texts: List[str],
    ) -> Tuple[List[Image.Image], List[str], Dict]:
        """Normalize extraction training data (Donut)."""
        logger.info("Normalizing extraction data...")
        
        # Normalize texts
        normalized_texts, text_indices = self.text_normalizer.normalize_batch(texts)
        
        # Filter images by valid text indices
        filtered_images = [images[i] for i in text_indices]
        
        # Normalize images
        normalized_images, image_indices = self.image_normalizer.normalize_batch(filtered_images)
        
        # Align both lists
        final_texts = [normalized_texts[i] for i in image_indices]
        final_images = normalized_images
        
        # Validation report
        report = self.validator.validate_dataset(final_texts, images=list(range(len(final_images))))
        report["stage"] = "extraction"
        report["original_samples"] = len(images)
        report["normalized_samples"] = len(final_images)
        
        logger.info(f"Extraction data normalization complete: {report}")
        
        return final_images, final_texts, report
    
    def normalize_ranking_data(
        self,
        resumes: List[str],
        job_descriptions: List[str],
        scores: List[float],
        labels: Optional[List[int]] = None,
    ) -> Tuple[List[str], List[str], List[float], List[int], Dict]:
        """Normalize ranking training data (BGE-M3)."""
        logger.info("Normalizing ranking data...")
        
        # Normalize texts
        normalized_resumes, resume_indices = self.text_normalizer.normalize_batch(resumes)
        
        # Filter JDs and scores
        filtered_jds = [job_descriptions[i] for i in resume_indices]
        filtered_scores = [scores[i] for i in resume_indices]
        if labels is not None:
            filtered_labels = [labels[i] for i in resume_indices]
        else:
            filtered_labels = None
        
        # Normalize JDs
        normalized_jds, jd_indices = self.text_normalizer.normalize_batch(filtered_jds)
        
        # Normalize scores
        normalized_scores = self.score_normalizer.normalize(
            [filtered_scores[i] for i in jd_indices]
        )
        
        # Align all lists
        final_resumes = [normalized_resumes[i] for i in jd_indices]
        final_jds = normalized_jds
        final_scores = normalized_scores
        
        if filtered_labels:
            final_labels = [filtered_labels[i] for i in jd_indices]
        else:
            final_labels = [1 if s > 0.5 else 0 for s in final_scores]
        
        # Remove outliers
        final_scores, score_indices = self.validator.detect_outliers(final_scores)
        final_resumes = [final_resumes[i] for i in score_indices]
        final_jds = [final_jds[i] for i in score_indices]
        final_labels = [final_labels[i] for i in score_indices]
        
        # Remove duplicates
        final_resumes, dup_indices = self.validator.remove_duplicates(final_resumes)
        final_jds = [final_jds[i] for i in dup_indices]
        final_scores = [final_scores[i] for i in dup_indices]
        final_labels = [final_labels[i] for i in dup_indices]
        
        # Balance data
        combined_data = list(zip(final_resumes, final_jds, final_scores))
        balanced_data, balanced_labels = self.balancer.balance(combined_data, final_labels)
        
        if balanced_data:
            final_resumes, final_jds, final_scores = zip(*balanced_data)
            final_resumes = list(final_resumes)
            final_jds = list(final_jds)
            final_scores = list(final_scores)
        
        # Validation report
        report = self.validator.validate_dataset(final_resumes, values=final_scores)
        report["stage"] = "ranking"
        report["original_samples"] = len(resumes)
        report["normalized_samples"] = len(final_resumes)
        
        logger.info(f"Ranking data normalization complete: {report}")
        
        return final_resumes, final_jds, final_scores, balanced_labels, report


def main():
    """Example usage."""
    config = NormalizationConfig(
        lowercase=True,
        remove_extra_whitespace=True,
        normalize_unicode=True,
        apply_augmentation=True,
        balance_data=True,
    )
    
    normalizer = DataNormalizer(config)
    
    # Example: Normalize ranking data
    resumes = [
        "  John  Doe  -   Python Developer   ",
        "Jane Smith - Senior Software Engineer",
    ]
    jds = [
        "We need a Python developer with 5+ years",
        "Looking for a senior software engineer",
    ]
    scores = [0.85, 0.92]
    
    r, j, s, l, report = normalizer.normalize_ranking_data(resumes, jds, scores)
    
    logger.info("Normalized data:")
    for res, jd, score in zip(r, j, s):
        logger.info(f"Resume: {res[:50]}...")
        logger.info(f"JD: {jd[:50]}...")
        logger.info(f"Score: {score}\n")


if __name__ == "__main__":
    main()
