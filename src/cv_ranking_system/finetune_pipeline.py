"""
Fine-tuning Pipeline Orchestrator for CV Ranking System.

Provides utilities for managing and executing the complete fine-tuning pipeline
including data preparation, model training, and evaluation.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum

import torch
import numpy as np
from tqdm import tqdm

from cv_ranking_system.utils.logging_utils import setup_logger
from cv_ranking_system.data_normalizer import (
    DataNormalizer, NormalizationConfig
)
from huggingface_hub import snapshot_download
import kaggle

logger = setup_logger(__name__)


class Stage(Enum):
    """Fine-tuning pipeline stages."""
    PREPARATION = "preparation"
    DONUT_TRAINING = "donut_training"
    BGE_TRAINING = "bge_training"
    EVALUATION = "evaluation"
    DEPLOYMENT = "deployment"


@dataclass
class PipelineConfig:
    """Main configuration for the fine-tuning pipeline."""
    # Directories
    root_dir: str = "./"
    data_dir: str = "./data"
    models_dir: str = "./models"
    logs_dir: str = "./logs"
    output_dir: str = "./outputs"
    
    # Stages to run
    run_donut: bool = True
    run_bge: bool = True
    run_evaluation: bool = True
    
    # Resource configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_gpus: int = torch.cuda.device_count()
    mixed_precision: bool = torch.cuda.is_available()
    
    # Experiment tracking
    experiment_name: str = "cv-ranking-finetuning"
    timestamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def get_run_id(self) -> str:
        """Get unique run identifier."""
        return f"{self.experiment_name}_{self.timestamp}"


class DatasetManager:
    """Manage dataset preparation and loading."""
    
    def __init__(self, config: PipelineConfig):
        """Initialize dataset manager."""
        self.config = config
        self.data_dir = Path(config.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized DatasetManager at {self.data_dir}")
    
    def download_datasets(self) -> Dict[str, str]:
        """Download datasets from sources."""
        logger.info("Downloading datasets...")
        
        datasets_info = {
            "resume_ner": "yashpwrr/resume-ner-training-dataset",
            "resume_jd_match": "facehuggerapoorv/resume-jd-match",
            "resume_score": "netsol/resume-score-details",
            "resumes": "datasetmaster/resumes",
        }
        
        for dataset_name, source in datasets_info.items():
            dataset_path = self.data_dir / dataset_name
            dataset_path.mkdir(parents=True, exist_ok=True)
            
            try:
                logger.info(f"Downloading {dataset_name}...")
                if dataset_name == "resume_ner":
                    kaggle.api.dataset_download_files(source, path=str(dataset_path), unzip=True)
                else:
                    snapshot_download(repo_id=source, repo_type="dataset", local_dir=str(dataset_path))
                logger.info(f"Successfully downloaded {dataset_name}")
            except Exception as e:
                logger.error(f"Failed to download {dataset_name}: {str(e)}")  
                          
        return datasets_info
    
    def prepare_training_data(
        self,
        dataset_type: str = "all",
        train_split: float = 0.8,
    ) -> Dict[str, Path]:
        """Prepare training data splits."""
        logger.info(f"Preparing training data (train_split={train_split})...")
        
        splits = {
            "train": self.data_dir / f"{dataset_type}_train",
            "val": self.data_dir / f"{dataset_type}_val",
            "test": self.data_dir / f"{dataset_type}_test",
        }
        
        for split_dir in splits.values():
            split_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created data splits: {list(splits.keys())}")
        
        return splits
    
    def validate_data_integrity(self) -> bool:
        """Validate downloaded data integrity."""
        logger.info("Validating data integrity...")
        
        required_dirs = [
            self.data_dir / "resume_ner",
            self.data_dir / "resume_jd_match",
        ]
        
        all_valid = True
        for data_dir in required_dirs:
            if data_dir.exists():
                file_count = len(list(data_dir.glob("*")))
                logger.info(f"{data_dir.name}: {file_count} files")
            else:
                logger.warning(f"{data_dir.name}: Not found")
                all_valid = False
        
        return all_valid
    
    def get_dataset_statistics(self) -> Dict:
        """Get statistics about loaded datasets."""
        stats = {
            "timestamp": datetime.now().isoformat(),
            "data_dir": str(self.data_dir),
            "datasets": {},
        }
        
        for dataset_dir in self.data_dir.iterdir():
            if dataset_dir.is_dir():
                file_count = len(list(dataset_dir.glob("*")))
                total_size = sum(f.stat().st_size for f in dataset_dir.glob("**/*"))
                
                stats["datasets"][dataset_dir.name] = {
                    "file_count": file_count,
                    "total_size_mb": total_size / (1024 * 1024),
                }
        
        return stats


class ModelRegistry:
    """Registry for managing fine-tuned models."""
    
    def __init__(self, config: PipelineConfig):
        """Initialize model registry."""
        self.config = config
        self.models_dir = Path(config.models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.registry_file = self.models_dir / "registry.json"
        self.registry = self._load_registry()
        
        logger.info(f"Initialized ModelRegistry at {self.models_dir}")
    
    def _load_registry(self) -> Dict:
        """Load model registry from file."""
        if self.registry_file.exists():
            with open(self.registry_file, "r") as f:
                return json.load(f)
        return {}
    
    def _save_registry(self):
        """Save model registry to file."""
        with open(self.registry_file, "w") as f:
            json.dump(self.registry, f, indent=2)
    
    def register_model(
        self,
        model_name: str,
        model_type: str,
        model_path: str,
        metadata: Dict,
    ):
        """Register a fine-tuned model."""
        entry = {
            "name": model_name,
            "type": model_type,
            "path": str(model_path),
            "registered_at": datetime.now().isoformat(),
            "metadata": metadata,
        }
        
        self.registry[model_name] = entry
        self._save_registry()
        
        logger.info(f"Registered model: {model_name}")
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get information about a registered model."""
        return self.registry.get(model_name)
    
    def list_models(self, model_type: Optional[str] = None) -> List[Dict]:
        """List all registered models."""
        models = list(self.registry.values())
        
        if model_type:
            models = [m for m in models if m["type"] == model_type]
        
        return models
    
    def get_best_model(self, model_type: str) -> Optional[Dict]:
        """Get the best performing model of a type."""
        models = self.list_models(model_type)
        
        if not models:
            return None
        
        # Sort by registered_at (latest first)
        models.sort(key=lambda m: m["registered_at"], reverse=True)
        
        return models[0]


class PipelineTracker:
    """Track pipeline execution progress and metadata."""
    
    def __init__(self, config: PipelineConfig):
        """Initialize pipeline tracker."""
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.run_id = config.get_run_id()
        self.run_dir = self.output_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata = {
            "run_id": self.run_id,
            "config": asdict(config),
            "started_at": datetime.now().isoformat(),
            "stages": {},
        }
        
        logger.info(f"Initialized PipelineTracker with run_id: {self.run_id}")
    
    def record_stage(self, stage: Stage, status: str, details: Dict = None):
        """Record stage execution."""
        self.metadata["stages"][stage.value] = {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "details": details or {},
        }
        
        logger.info(f"Stage {stage.value}: {status}")
        self._save_metadata()
    
    def record_model_checkpoint(self, stage: Stage, checkpoint_info: Dict):
        """Record model checkpoint information."""
        if stage.value not in self.metadata["stages"]:
            self.metadata["stages"][stage.value] = {}
        
        self.metadata["stages"][stage.value]["checkpoint"] = checkpoint_info
        self._save_metadata()
    
    def record_metrics(self, stage: Stage, metrics: Dict):
        """Record stage metrics."""
        if stage.value not in self.metadata["stages"]:
            self.metadata["stages"][stage.value] = {}
        
        self.metadata["stages"][stage.value]["metrics"] = metrics
        self._save_metadata()
    
    def _save_metadata(self):
        """Save metadata to file."""
        metadata_file = self.run_dir / "metadata.json"
        
        # Convert datetime objects to strings for serialization
        serializable_metadata = json.loads(
            json.dumps(self.metadata, default=str)
        )
        
        with open(metadata_file, "w") as f:
            json.dump(serializable_metadata, f, indent=2)
    
    def get_summary(self) -> Dict:
        """Get pipeline execution summary."""
        self.metadata["completed_at"] = datetime.now().isoformat()
        
        return self.metadata


class FineTuningOrchestrator:
    """Orchestrate the complete fine-tuning pipeline."""
    
    def __init__(self, config: PipelineConfig):
        """Initialize orchestrator."""
        self.config = config
        
        # Initialize components
        self.dataset_manager = DatasetManager(config)
        self.model_registry = ModelRegistry(config)
        self.tracker = PipelineTracker(config)
        
        # Initialize data normalizer
        norm_config = NormalizationConfig(
            lowercase=True,
            remove_extra_whitespace=True,
            normalize_unicode=True,
            apply_augmentation=False,
            balance_data=True,
            detect_outliers=True,
        )
        self.data_normalizer = DataNormalizer(norm_config)
        
        # Setup directories
        Path(config.logs_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized FineTuningOrchestrator")
    
    def setup_environment(self):
        """Setup training environment."""
        logger.info("Setting up environment...")
        
        # Check GPU availability
        if self.config.device == "cuda":
            logger.info(f"GPUs available: {self.config.num_gpus}")
            for i in range(self.config.num_gpus):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"  GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
        
        # Set environment variables
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        
        self.tracker.record_stage(
            Stage.PREPARATION,
            "completed",
            {"gpus": self.config.num_gpus, "device": self.config.device}
        )
    
    def prepare_data(self):
        """Prepare and normalize data for training."""
        logger.info("Preparing and normalizing data...")
        
        try:
            # Validate existing data
            is_valid = self.dataset_manager.validate_data_integrity()
            
            if not is_valid:
                logger.info("Downloading datasets...")
                self.dataset_manager.download_datasets()
            
            # Prepare splits
            splits = self.dataset_manager.prepare_training_data()
            
            # Get statistics
            stats = self.dataset_manager.get_dataset_statistics()
            
            # Normalize data
            normalization_report = self._normalize_datasets()
            stats["normalization"] = normalization_report
            
            self.tracker.record_stage(
                Stage.PREPARATION,
                "completed",
                stats
            )
            
            logger.info("Data preparation and normalization completed")
        
        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            self.tracker.record_stage(Stage.PREPARATION, "failed", {"error": str(e)})
            raise
    
    def _normalize_datasets(self) -> Dict:
        """Normalize all datasets using data normalizer."""
        logger.info("Starting data normalization...")
        
        normalization_report = {
            "extraction_data": {},
            "ranking_data": {},
        }
        
        try:
            # TODO: Implement normalization for extraction data (images + texts)
            # This would require loading the actual datasets and calling:
            # images, texts, report = self.data_normalizer.normalize_extraction_data(...)
            logger.info("Extraction data normalization would process images and texts")
            
            # TODO: Implement normalization for ranking data (resumes + JDs + scores)
            # This would require loading the actual datasets and calling:
            # resumes, jds, scores, labels, report = self.data_normalizer.normalize_ranking_data(...)
            logger.info("Ranking data normalization would process resume-JD pairs")
            
            logger.info("Data normalization completed")
        
        except Exception as e:
            logger.error(f"Data normalization failed: {e}")
            normalization_report["error"] = str(e)
        
        return normalization_report
    
    def train_donut(self):
        """Train Donut extraction model."""
        if not self.config.run_donut:
            logger.info("Skipping Donut training")
            return
        
        logger.info("Starting Donut training...")
        
        try:
            from cv_ranking_system.extraction.finetune_donut import (
                DonutFineTuner, DonutConfig
            )
            
            # Configuration
            donut_config = DonutConfig(
                output_dir=str(Path(self.config.models_dir) / "donut-finetuned"),
                device=self.config.device,
            )
            
            # Train
            trainer = DonutFineTuner(donut_config)
            model, processor = trainer.train()
            
            # Register model
            self.model_registry.register_model(
                model_name=f"donut-{self.tracker.run_id}",
                model_type="extraction",
                model_path=donut_config.output_dir,
                metadata=asdict(donut_config),
            )
            
            self.tracker.record_stage(
                Stage.DONUT_TRAINING,
                "completed",
                {"model_dir": donut_config.output_dir}
            )
            
            logger.info("Donut training completed")
        
        except Exception as e:
            logger.error(f"Donut training failed: {e}")
            self.tracker.record_stage(Stage.DONUT_TRAINING, "failed", {"error": str(e)})
            raise
    
    def train_bge(self):
        """Train BGE-M3 ranking model."""
        if not self.config.run_bge:
            logger.info("Skipping BGE training")
            return
        
        logger.info("Starting BGE-M3 training...")
        
        try:
            from cv_ranking_system.retrieval.finetune_bge import (
                BGEFineTuner, BGEConfig
            )
            
            # Configuration
            bge_config = BGEConfig(
                output_dir=str(Path(self.config.models_dir) / "bge-m3-finetuned"),
                device=self.config.device,
            )
            
            # Train
            trainer = BGEFineTuner(bge_config)
            model = trainer.train()
            
            # Register model
            self.model_registry.register_model(
                model_name=f"bge-m3-{self.tracker.run_id}",
                model_type="ranking",
                model_path=bge_config.output_dir,
                metadata=asdict(bge_config),
            )
            
            self.tracker.record_stage(
                Stage.BGE_TRAINING,
                "completed",
                {"model_dir": bge_config.output_dir}
            )
            
            logger.info("BGE-M3 training completed")
        
        except Exception as e:
            logger.error(f"BGE training failed: {e}")
            self.tracker.record_stage(Stage.BGE_TRAINING, "failed", {"error": str(e)})
            raise
    
    def evaluate_models(self):
        """Evaluate fine-tuned models."""
        if not self.config.run_evaluation:
            logger.info("Skipping evaluation")
            return
        
        logger.info("Starting model evaluation...")
        
        try:
            from tests.test_finetuned_models import (
                ModelEvaluator, EvaluationConfig
            )
            
            eval_config = EvaluationConfig(
                output_dir=str(self.tracker.run_dir / "evaluation"),
                device=self.config.device,
            )
            
            evaluator = ModelEvaluator(eval_config)
            results = evaluator.run_evaluation()
            
            self.tracker.record_stage(
                Stage.EVALUATION,
                "completed",
                {"output_dir": eval_config.output_dir}
            )
            
            logger.info("Model evaluation completed")
        
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            self.tracker.record_stage(Stage.EVALUATION, "failed", {"error": str(e)})
    
    def run_pipeline(self):
        """Execute the complete fine-tuning pipeline."""
        try:
            logger.info(f"Starting fine-tuning pipeline (run_id: {self.tracker.run_id})")
            
            # Setup
            self.setup_environment()
            self.prepare_data()
            
            # Training
            self.train_donut()
            self.train_bge()
            
            # Evaluation
            self.evaluate_models()
            
            # Summary
            summary = self.tracker.get_summary()
            
            logger.info("\n" + "="*60)
            logger.info("PIPELINE SUMMARY")
            logger.info("="*60)
            logger.info(json.dumps(summary, indent=2, default=str))
            
            logger.info("Fine-tuning pipeline completed successfully!")
            
            return summary
        
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise


def main():
    """Main orchestration script."""
    # Create configuration
    config = PipelineConfig(
        run_donut=True,
        run_bge=True,
        run_evaluation=True,
    )
    
    # Run pipeline
    orchestrator = FineTuningOrchestrator(config)
    summary = orchestrator.run_pipeline()


if __name__ == "__main__":
    main()
