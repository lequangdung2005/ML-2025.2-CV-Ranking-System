"""
Dataset Management Utilities for Fine-Tuning Pipeline

Handles downloading, preparing, and managing datasets from various sources:
- Kaggle datasets
- HuggingFace datasets
- Local datasets
- Web-based datasets
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import shutil
import subprocess
from urllib.parse import urlparse

import requests
from tqdm import tqdm

from cv_ranking_system.utils.logging_utils import setup_logger
from cv_ranking_system.credentials import (
    get_credentials_manager,
    initialize_credentials,
)

logger = setup_logger(__name__)


class DatasetSource(Enum):
    """Supported dataset sources."""
    KAGGLE = "kaggle"
    HUGGINGFACE = "huggingface"
    LOCAL = "local"
    WEB = "web"
    GITHUB = "github"


@dataclass
class DatasetInfo:
    """Information about a dataset."""
    name: str
    source: DatasetSource
    url: str
    description: str
    size_gb: float
    train_size: int
    val_size: Optional[int] = None
    test_size: Optional[int] = None


class DatasetCatalog:
    """Catalog of available datasets for fine-tuning."""
    
    DATASETS: Dict[str, DatasetInfo] = {
        # Extraction/OCR datasets
        "resume_ner_kaggle": DatasetInfo(
            name="Resume NER Training Dataset",
            source=DatasetSource.KAGGLE,
            url="yashpwrr/resume-ner-training-dataset",
            description="Named Entity Recognition dataset for resume text extraction",
            size_gb=0.5,
            train_size=2000,
            val_size=500,
            test_size=500,
        ),
        "resume_corpus_github": DatasetInfo(
            name="Resume Corpus Dataset",
            source=DatasetSource.GITHUB,
            url="vrundag91/Resume-Corpus-Dataset",
            description="Corpus of resume documents for extraction training",
            size_gb=1.0,
            train_size=5000,
        ),
        
        # Ranking/Retrieval datasets
        "resume_jd_match": DatasetInfo(
            name="Resume-JD Matching Dataset",
            source=DatasetSource.HUGGINGFACE,
            url="facehuggerapoorv/resume-jd-match",
            description="Resume and job description matching pairs with scores",
            size_gb=0.2,
            train_size=3000,
            val_size=500,
            test_size=500,
        ),
        "resume_score": DatasetInfo(
            name="Resume Score Details Dataset",
            source=DatasetSource.HUGGINGFACE,
            url="netsol/resume-score-details",
            description="Resume-JD pairs with detailed scoring",
            size_gb=0.15,
            train_size=2000,
            val_size=300,
        ),
        
        # General resume datasets
        "resumes_hf": DatasetInfo(
            name="Resumes Dataset",
            source=DatasetSource.HUGGINGFACE,
            url="datasetmaster/resumes",
            description="General collection of resume documents",
            size_gb=0.5,
            train_size=1000,
        ),
        "it_resumes": DatasetInfo(
            name="IT Resume Dataset",
            source=DatasetSource.WEB,
            url="https://cvparserpro.io/it-resume-dataset",
            description="IT-focused resume documents",
            size_gb=0.3,
            train_size=500,
        ),
    }
    
    @classmethod
    def get_dataset_info(cls, dataset_key: str) -> Optional[DatasetInfo]:
        """Get information about a dataset."""
        return cls.DATASETS.get(dataset_key)
    
    @classmethod
    def list_datasets(cls, source: Optional[DatasetSource] = None) -> List[str]:
        """List available datasets, optionally filtered by source."""
        datasets = cls.DATASETS.keys()
        
        if source:
            datasets = [
                key for key, info in cls.DATASETS.items()
                if info.source == source
            ]
        
        return list(datasets)
    
    @classmethod
    def get_total_size(cls, dataset_keys: List[str]) -> float:
        """Get total size of multiple datasets."""
        total = 0
        for key in dataset_keys:
            info = cls.get_dataset_info(key)
            if info:
                total += info.size_gb
        return total


class DatasetDownloader:
    """Download datasets from various sources."""
    
    def __init__(self, output_dir: Path, credentials_manager=None):
        """Initialize downloader.
        
        Args:
            output_dir: Directory to download datasets to.
            credentials_manager: Custom credentials manager. If None, uses default.
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize credentials manager
        self.credentials_manager = credentials_manager or get_credentials_manager()
        
        # Setup authentication
        self.credentials_manager.setup_all_auth()
    
    def download_kaggle_dataset(self, dataset_id: str) -> Path:
        """Download dataset from Kaggle."""
        logger.info(f"Downloading Kaggle dataset: {dataset_id}")
        
        try:
            import kaggle
        except ImportError:
            logger.error("kaggle package not installed. Install with: pip install kaggle")
            logger.info("Setup Kaggle API: https://github.com/Kaggle/kaggle-api#api-credentials")
            raise
        
        # Validate Kaggle credentials
        if not self.credentials_manager.validate_kaggle():
            status = self.credentials_manager.get_status()
            logger.error(f"Kaggle credentials status: {status['kaggle']}")
            raise ValueError(
                "Kaggle credentials not configured. "
                "Please set KAGGLE_USERNAME and KAGGLE_API_KEY in .env file or environment variables."
            )
        
        dataset_dir = self.output_dir / dataset_id.split("/")[-1]
        
        try:
            kaggle.api.dataset_download_files(
                dataset_id,
                path=str(dataset_dir),
                unzip=True,
            )
            logger.info(f"Downloaded to {dataset_dir}")
            return dataset_dir
        
        except Exception as e:
            logger.error(f"Failed to download Kaggle dataset: {e}")
            raise
    
    def download_huggingface_dataset(self, dataset_id: str) -> Path:
        """Download dataset from HuggingFace."""
        logger.info(f"Downloading HuggingFace dataset: {dataset_id}")
        
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
        except ImportError:
            logger.error("huggingface_hub not installed. Install with: pip install huggingface_hub")
            raise
        
        # Validate HuggingFace credentials
        if not self.credentials_manager.validate_huggingface():
            status = self.credentials_manager.get_status()
            logger.error(f"HuggingFace credentials status: {status['huggingface']}")
            logger.warning(
                "HuggingFace authentication failed. Some datasets may require authentication. "
                "Please set HUGGINGFACE_API_TOKEN in .env file or environment variables."
            )
        
        dataset_name = dataset_id.split("/")[-1]
        dataset_dir = self.output_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # List files in the dataset
            files = list_repo_files(dataset_id, repo_type="dataset")
            logger.info(f"Found {len(files)} files in dataset")
            
            # Download files
            for file_path in tqdm(files[:10], desc="Downloading files"):  # Limit to first 10
                hf_hub_download(
                    repo_id=dataset_id,
                    filename=file_path,
                    repo_type="dataset",
                    local_dir=str(dataset_dir),
                )
            
            logger.info(f"Downloaded to {dataset_dir}")
            return dataset_dir
        
        except Exception as e:
            logger.error(f"Failed to download HuggingFace dataset: {e}")
            raise
    
    def download_github_dataset(self, repo_url: str) -> Path:
        """Clone dataset from GitHub."""
        logger.info(f"Cloning GitHub repository: {repo_url}")
        
        repo_name = repo_url.split("/")[-1]
        repo_dir = self.output_dir / repo_name
        
        if repo_dir.exists():
            logger.info(f"Repository already exists at {repo_dir}")
            return repo_dir
        
        try:
            git_url = f"https://github.com/{repo_url}.git"
            subprocess.run(
                ["git", "clone", git_url, str(repo_dir)],
                check=True,
                capture_output=True,
            )
            logger.info(f"Cloned to {repo_dir}")
            return repo_dir
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone repository: {e}")
            raise
    
    def download_web_dataset(self, url: str, filename: Optional[str] = None) -> Path:
        """Download dataset from web URL."""
        logger.info(f"Downloading from URL: {url}")
        
        if not filename:
            filename = urlparse(url).path.split("/")[-1]
        
        file_path = self.output_dir / filename
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get("content-length", 0))
            
            with open(file_path, "wb") as f:
                with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            logger.info(f"Downloaded to {file_path}")
            return file_path
        
        except Exception as e:
            logger.error(f"Failed to download: {e}")
            raise
    
    def download_dataset(self, dataset_key: str) -> Path:
        """Download a dataset by key from the catalog."""
        info = DatasetCatalog.get_dataset_info(dataset_key)
        
        if not info:
            raise ValueError(f"Unknown dataset: {dataset_key}")
        
        logger.info(f"Starting download: {info.name}")
        logger.info(f"  Size: {info.size_gb} GB")
        logger.info(f"  Samples: {info.train_size + (info.val_size or 0) + (info.test_size or 0)}")
        
        try:
            if info.source == DatasetSource.KAGGLE:
                return self.download_kaggle_dataset(info.url)
            
            elif info.source == DatasetSource.HUGGINGFACE:
                return self.download_huggingface_dataset(info.url)
            
            elif info.source == DatasetSource.GITHUB:
                return self.download_github_dataset(info.url)
            
            elif info.source == DatasetSource.WEB:
                return self.download_web_dataset(info.url)
            
            else:
                raise ValueError(f"Unsupported source: {info.source}")
        
        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise


class DatasetValidator:
    """Validate downloaded datasets."""
    
    @staticmethod
    def validate_dataset(dataset_dir: Path) -> Dict:
        """Validate a downloaded dataset."""
        logger.info(f"Validating dataset at {dataset_dir}")
        
        if not dataset_dir.exists():
            raise ValueError(f"Dataset directory not found: {dataset_dir}")
        
        validation_result = {
            "path": str(dataset_dir),
            "exists": True,
            "file_count": 0,
            "total_size_mb": 0,
            "file_types": {},
            "samples": {},
        }
        
        # Count files and sizes
        for file_path in dataset_dir.rglob("*"):
            if file_path.is_file():
                validation_result["file_count"] += 1
                validation_result["total_size_mb"] += file_path.stat().st_size / (1024 * 1024)
                
                # Track file types
                suffix = file_path.suffix.lower()
                validation_result["file_types"][suffix] = validation_result["file_types"].get(suffix, 0) + 1
        
        logger.info(f"  Files: {validation_result['file_count']}")
        logger.info(f"  Total size: {validation_result['total_size_mb']:.2f} MB")
        logger.info(f"  File types: {validation_result['file_types']}")
        
        return validation_result
    
    @staticmethod
    def validate_integrity(dataset_dir: Path, expected_files: List[str]) -> bool:
        """Validate that dataset contains expected files."""
        logger.info(f"Checking for expected files...")
        
        missing_files = []
        for filename in expected_files:
            file_path = dataset_dir / filename
            if not file_path.exists():
                missing_files.append(filename)
        
        if missing_files:
            logger.warning(f"Missing files: {missing_files}")
            return False
        
        logger.info("All expected files found")
        return True


class DatasetManager:
    """Main dataset management interface."""
    
    def __init__(self, root_dir: str = "./data", credentials_manager=None):
        """Initialize manager.
        
        Args:
            root_dir: Root directory for downloaded datasets.
            credentials_manager: Custom credentials manager. If None, uses default.
        """
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        
        self.credentials_manager = credentials_manager or get_credentials_manager()
        self.downloader = DatasetDownloader(self.root_dir, self.credentials_manager)
        self.validator = DatasetValidator()
        
        logger.info(f"DatasetManager initialized at {self.root_dir}")
        logger.info(f"Credentials status: {self.credentials_manager.get_status()}")
    
    def download_extraction_datasets(self) -> Dict[str, Path]:
        """Download all extraction-related datasets."""
        logger.info("Downloading extraction datasets...")
        
        datasets = {
            "resume_ner": "resume_ner_kaggle",
            "resume_corpus": "resume_corpus_github",
        }
        
        downloaded = {}
        
        for name, key in datasets.items():
            try:
                path = self.downloader.download_dataset(key)
                self.validator.validate_dataset(path)
                downloaded[name] = path
            except Exception as e:
                logger.error(f"Failed to download {name}: {e}")
        
        return downloaded
    
    def download_ranking_datasets(self) -> Dict[str, Path]:
        """Download all ranking-related datasets."""
        logger.info("Downloading ranking datasets...")
        
        datasets = {
            "resume_jd_match": "resume_jd_match",
            "resume_score": "resume_score",
        }
        
        downloaded = {}
        
        for name, key in datasets.items():
            try:
                path = self.downloader.download_dataset(key)
                self.validator.validate_dataset(path)
                downloaded[name] = path
            except Exception as e:
                logger.error(f"Failed to download {name}: {e}")
        
        return downloaded
    
    def download_all(self) -> Dict[str, Path]:
        """Download all recommended datasets."""
        logger.info("Downloading all datasets...")
        
        all_downloaded = {}
        all_downloaded.update(self.download_extraction_datasets())
        all_downloaded.update(self.download_ranking_datasets())
        
        return all_downloaded
    
    def get_dataset_info(self) -> Dict:
        """Get information about all datasets."""
        info = {
            "root_dir": str(self.root_dir),
            "catalog": {
                key: {
                    "name": dataset.name,
                    "source": dataset.source.value,
                    "size_gb": dataset.size_gb,
                    "total_samples": dataset.train_size + (dataset.val_size or 0) + (dataset.test_size or 0),
                }
                for key, dataset in DatasetCatalog.DATASETS.items()
            },
            "downloaded": {},
        }
        
        # Check which datasets are already downloaded
        for dataset_dir in self.root_dir.iterdir():
            if dataset_dir.is_dir():
                validation = self.validator.validate_dataset(dataset_dir)
                info["downloaded"][dataset_dir.name] = validation
        
        return info


def main():
    """Main script for dataset management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dataset management utility")
    parser.add_argument(
        "action",
        choices=["list", "download", "validate", "info"],
        help="Action to perform",
    )
    parser.add_argument(
        "--dataset",
        help="Specific dataset to download or validate",
    )
    parser.add_argument(
        "--output-dir",
        default="./data",
        help="Output directory for datasets",
    )
    parser.add_argument(
        "--extraction-only",
        action="store_true",
        help="Only download extraction datasets",
    )
    parser.add_argument(
        "--ranking-only",
        action="store_true",
        help="Only download ranking datasets",
    )
    
    args = parser.parse_args()
    
    manager = DatasetManager(args.output_dir)
    
    if args.action == "list":
        datasets = DatasetCatalog.list_datasets()
        logger.info("Available datasets:")
        for dataset_key in datasets:
            info = DatasetCatalog.get_dataset_info(dataset_key)
            logger.info(f"  - {dataset_key}: {info.name} ({info.size_gb} GB)")
    
    elif args.action == "download":
        if args.dataset:
            logger.info(f"Downloading {args.dataset}...")
            manager.downloader.download_dataset(args.dataset)
        elif args.extraction_only:
            manager.download_extraction_datasets()
        elif args.ranking_only:
            manager.download_ranking_datasets()
        else:
            manager.download_all()
    
    elif args.action == "validate":
        dataset_dir = Path(args.output_dir) / args.dataset
        if dataset_dir.exists():
            manager.validator.validate_dataset(dataset_dir)
        else:
            logger.error(f"Dataset not found: {dataset_dir}")
    
    elif args.action == "info":
        info = manager.get_dataset_info()
        logger.info(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
