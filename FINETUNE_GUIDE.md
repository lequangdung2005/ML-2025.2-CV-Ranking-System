# Fine-Tuning Pipeline Guide

This guide covers the fine-tuning scripts for the CV Ranking System, including model training and evaluation.

## Overview

The fine-tuning pipeline consists of three main components:

1. **Donut Fine-tuning** (`finetune_donut.py`) - Document Understanding Transformer for OCR/extraction
2. **BGE-M3 Fine-tuning** (`finetune_bge.py`) - BAAI General Embedding for ranking/retrieval
3. **Model Evaluation** (`test_finetuned_models.py`) - Comprehensive testing and benchmarking
4. **Pipeline Orchestrator** (`finetune_pipeline.py`) - Unified management and execution

## Prerequisites

### Dependencies

```bash
pip install torch torchvision transformers datasets pillow scikit-learn
pip install sentence-transformers scipy matplotlib seaborn
```

### GPU Requirements

- CUDA 11.8+ (recommended)
- GPU with minimum 8GB VRAM (16GB+ recommended)
- PyTorch with CUDA support

## Stage 1: Ingestion & Extraction (Donut)

### Configuration

```python
from cv_ranking_system.extraction.finetune_donut import DonutConfig, DonutFineTuner

config = DonutConfig(
    model_name="naver-clova-ix/donut-base",
    output_dir="./models/donut-finetuned",
    num_epochs=10,
    batch_size=8,
    learning_rate=1e-4,
    use_kaggle_dataset=True,
    use_huggingface_dataset=True,
)
```

### Available Datasets

1. **Resume NER Training Dataset** (Kaggle)
   - URL: https://www.kaggle.com/datasets/yashpwrr/resume-ner-training-dataset
   - Setup: `kaggle datasets download -d yashpwrr/resume-ner-training-dataset`

2. **Resume Corpus Dataset** (GitHub)
   - URL: https://github.com/vrundag91/Resume-Corpus-Dataset
   - Clone and place in `./data/resume-corpus/`

### Training

```python
fine_tuner = DonutFineTuner(config)
model, processor = fine_tuner.train()
```

### Key Features

- Multi-dataset support (Kaggle + HuggingFace)
- Automatic train/val splitting (90/10)
- Early stopping with configurable patience
- Mixed precision training support
- Checkpoint saving and recovery

### Expected Output

```
Output Directory: ./models/donut-finetuned/
├── pytorch_model.bin         # Fine-tuned model weights
├── config.json              # Model config
├── processor_config.json    # Processor config
└── finetune_config.json     # Training configuration
```

## Stage 2: Ranking & Retrieval (BGE-M3)

### Configuration

```python
from cv_ranking_system.retrieval.finetune_bge import BGEConfig, BGEFineTuner

config = BGEConfig(
    model_name="BAAI/bge-m3",
    output_dir="./models/bge-m3-finetuned",
    num_epochs=5,
    batch_size=32,
    learning_rate=2e-5,
    loss_type="contrastive",  
    use_sentence_transformers=True,
)
```

### Available Datasets

1. **Resume-JD Matching** (HuggingFace)
   - URL: https://huggingface.co/datasets/facehuggerapoorv/resume-jd-match
   - Automatically loaded via HuggingFace datasets library

2. **Resume Score Details** (HuggingFace)
   - URL: https://huggingface.co/datasets/netsol/resume-score-details
   - Automatically loaded via HuggingFace datasets library

### Training

```python
fine_tuner = BGEFineTuner(config)
model = fine_tuner.train()
```

### Two Training Modes

#### Mode 1: Sentence Transformers (Recommended)

```python
config.use_sentence_transformers = True
```

Advantages:
- Simpler API
- Built-in loss functions (CosineSimilarity, Triplet, etc.)
- Automatic validation
- Better defaults

#### Mode 2: Raw Transformers

```python
config.use_sentence_transformers = False
```

Advantages:
- More control over training loop
- Custom loss functions
- Lower memory overhead

### Loss Functions

1. **Contrastive Loss** (default)
   - Best for: Similarity-based matching
   - Use case: Ranking resumes by relevance score

2. **Triplet Loss**
   - Best for: Hard negative mining
   - Use case: Learning fine-grained distinctions

### Expected Output

```
Output Directory: ./models/bge-m3-finetuned/
├── pytorch_model.bin         # Fine-tuned model weights
├── config.json              # Model config
├── tokenizer.json           # Tokenizer config
└── training_results.json    # Training metrics
```

## Stage 3: Testing & Evaluation

### Running Evaluations

```python
from tests.test_finetuned_models import ModelEvaluator, EvaluationConfig

config = EvaluationConfig(
    donut_model_path="./models/donut-finetuned",
    bge_model_path="./models/bge-m3-finetuned",
    output_dir="./evaluation_results",
    generate_plots=True,
)

evaluator = ModelEvaluator(config)
results = evaluator.run_evaluation()
```

### Evaluation Metrics

#### Donut Extraction

- Exact Match Rate: Percentage of perfectly extracted texts
- Partial Match Rate: Percentage of partially correct extractions
- Character Error Rate: Levenshtein distance-based metric

#### BGE-M3 Ranking

- **Regression Metrics:**
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)

- **Correlation Metrics:**
  - Spearman Correlation: Rank correlation
  - Pearson Correlation: Linear correlation

- **Ranking Metrics:**
  - MRR (Mean Reciprocal Rank)
  - NDCG@K (Normalized Discounted Cumulative Gain)

### Synthetic Test Data

The evaluation script generates synthetic data for testing:

```python
from tests.test_finetuned_models import SyntheticDataGenerator

# Generate synthetic resumes
resumes = SyntheticDataGenerator.generate_resume_samples(100)

# Generate synthetic job descriptions
jds = SyntheticDataGenerator.generate_jd_samples(20)
```

### Expected Output

```
Output Directory: ./evaluation_results/
├── evaluation_results.json   # Detailed metrics
├── bge_predictions.png       # Scatter plot of predictions vs true
├── bge_metrics.png           # Metrics bar chart
└── donut_extractions.json    # Sample extractions
```

## Stage 4: Pipeline Orchestration

### Full Pipeline Execution

```python
from cv_ranking_system.finetune_pipeline import (
    FineTuningOrchestrator, PipelineConfig
)

config = PipelineConfig(
    run_donut=True,
    run_bge=True,
    run_evaluation=True,
    experiment_name="cv-ranking-v1",
)

orchestrator = FineTuningOrchestrator(config)
summary = orchestrator.run_pipeline()
```

### Pipeline Components

1. **DatasetManager**
   - Download datasets
   - Prepare splits
   - Validate integrity
   - Generate statistics

2. **ModelRegistry**
   - Register trained models
   - Track model metadata
   - Retrieve best models
   - Version management

3. **PipelineTracker**
   - Record stage progress
   - Track metrics
   - Generate summaries
   - Persist metadata

### Execution Flow

```
1. Setup Environment
   ├── Check GPU availability
   ├── Set environment variables
   └── Initialize directories

2. Prepare Data
   ├── Download datasets
   ├── Validate integrity
   └── Create train/val/test splits

3. Train Donut
   ├── Load model
   ├── Prepare datasets
   ├── Fine-tune
   └── Register model

4. Train BGE-M3
   ├── Load model
   ├── Prepare datasets
   ├── Fine-tune
   └── Register model

5. Evaluate Models
   ├── Evaluate Donut
   ├── Evaluate BGE-M3
   ├── Generate plots
   └── Save results
```

## Command Line Usage

### Quick Start

```bash
# Train both models with orchestrator
python -m cv_ranking_system.finetune_pipeline

# Train only Donut
python -m cv_ranking_system.extraction.finetune_donut

# Train only BGE-M3
python -m cv_ranking_system.retrieval.finetune_bge

# Evaluate models
python -m pytest tests/test_finetuned_models.py -v
```

### Custom Configuration

Create a configuration file (`config.yaml`):

```yaml
donut:
  num_epochs: 10
  batch_size: 8
  learning_rate: 1e-4

bge:
  num_epochs: 5
  batch_size: 32
  learning_rate: 2e-5
  loss_type: contrastive

evaluation:
  generate_plots: true
  batch_size: 16
```

## Advanced Usage

### Custom Datasets

#### Adding Local Dataset (Donut)

```python
config = DonutConfig(
    local_data_path="./data/my_resumes",
)
# Expects structure:
# ./data/my_resumes/
#   ├── images/
#   │   ├── resume_1.png
#   │   └── resume_2.png
#   └── annotations.jsonl
```

Each line in `annotations.jsonl`:
```json
{
  "image_name": "resume_1.png",
  "extracted_text": "John Doe\nSoftware Engineer\n..."
}
```

#### Adding Local Dataset (BGE-M3)

```python
config = BGEConfig(
    local_data_path="./data/resume_jd_pairs.jsonl",
)
# Each line:
```json
{
  "resume": "John Doe, Python Developer...",
  "job_description": "We seek a Python developer...",
  "score": 0.85
}
```

### Distributed Training

For multi-GPU training:

```python
config = DonutConfig(
    device="cuda",
)

# With Hugging Face Trainer (already configured):
# Automatic detection and usage of multiple GPUs
```

### Mixed Precision Training

```python
config.use_mixed_precision = True  # Reduces memory usage by ~50%
```

### Checkpoint Recovery

Models automatically save checkpoints:

```python
# Resume from checkpoint
fine_tuner = DonutFineTuner(config)
# Loads best checkpoint automatically
model, processor = fine_tuner.train()
```

## Troubleshooting

### Out of Memory

```python
# Reduce batch size
config.batch_size = 4

# Enable gradient accumulation
config.gradient_accumulation_steps = 4

# Use mixed precision
config.mixed_precision = True
```

### Slow Training

```python
# Check GPU utilization
nvidia-smi

# Increase number of workers
config.num_workers = 4

# Use fp16
config.fp16 = True
```

### Dataset Not Found

```bash
# Ensure datasets are downloaded
python -c "
from huggingface_hub import list_repo_files
print(list_repo_files('facehuggerapoorv/resume-jd-match'))
"
```

## Model Deployment

### Save Model

```python
model.save_pretrained("./models/production/donut-v1")
processor.save_pretrained("./models/production/donut-v1")
```

### Load Model

```python
from transformers import AutoModel, AutoProcessor

model = AutoModel.from_pretrained("./models/production/donut-v1")
processor = AutoProcessor.from_pretrained("./models/production/donut-v1")
```

### Integration with System

```python
from cv_ranking_system.extraction.extract import DocumentExtractor

# Use fine-tuned model
extractor = DocumentExtractor(
    model_path="./models/production/donut-v1",
    use_fine_tuned=True,
)

text = extractor.extract("resume.pdf")
```

## Performance Benchmarks

### Donut

- Training time: ~2-4 hours (10 epochs, 1000 samples, V100)
- Inference time: ~0.5s per document
- Memory: ~10GB VRAM

### BGE-M3

- Training time: ~1-2 hours (5 epochs, 5000 pairs, V100)
- Inference time: ~10ms per text (batch size 32)
- Memory: ~8GB VRAM

## References

- [Donut Paper](https://arxiv.org/abs/2111.15664)
- [BGE-M3](https://huggingface.co/BAAI/bge-m3)
- [Hugging Face Documentation](https://huggingface.co/docs)
- [Sentence Transformers](https://www.sbert.net/)

## Support

For issues or questions:
1. Check logs in `./logs/` directory
2. Review pipeline metadata in `./{run_id}/metadata.json`
3. Enable verbose logging in configuration
