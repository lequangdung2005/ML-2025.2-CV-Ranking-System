# Data Normalization Guide

## Overview

Module `data_normalizer.py` cung cấp các chức năng chuẩn hóa dữ liệu toàn diện cho pipeline fine-tuning:

- **Text Normalization**: Làm sạch và chuẩn hóa văn bản
- **Image Normalization**: Tiền xử lý hình ảnh (resize, augmentation)
- **Score Normalization**: Chuẩn hóa điểm số về khoảng [0,1]
- **Data Validation**: Kiểm tra chất lượng dữ liệu
- **Outlier Detection**: Phát hiện và loại bỏ outliers
- **Data Balancing**: Cân bằng dữ liệu giữa các lớp

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Text Normalization (Chuẩn hóa văn bản)

```python
from cv_ranking_system.data_normalizer import (
    DataNormalizer, NormalizationConfig
)

# Tạo cấu hình
config = NormalizationConfig(
    lowercase=True,
    remove_extra_whitespace=True,
    normalize_unicode=True,
    min_text_length=10,
    max_text_length=10000,
)

# Khởi tạo normalizer
normalizer = DataNormalizer(config)

# Chuẩn hóa danh sách văn bản
texts = [
    "  John  Doe  -   Software   Engineer  ",
    "Jane Smith - Data Scientist",
    "Short",  # Will be filtered (too short)
]

# Batch normalization
normalized_texts, valid_indices = normalizer.text_normalizer.normalize_batch(texts)

print(f"Original: {len(texts)} texts")
print(f"After normalization: {len(normalized_texts)} valid texts")
print(f"Valid indices: {valid_indices}")
```

**Output:**
```
Original: 3 texts
After normalization: 2 valid texts
Valid indices: [0, 1]
```

### 2. Image Normalization (Chuẩn hóa hình ảnh)

```python
from PIL import Image

# Cấu hình với augmentation
config = NormalizationConfig(
    target_image_size=(1280, 960),
    image_format="RGB",
    apply_augmentation=True,  # Bật augmentation
)

normalizer = DataNormalizer(config)

# Normalize single image
image_path = "resume.png"
normalized_image = normalizer.image_normalizer.normalize(image_path)

# Batch normalization
image_paths = ["resume1.png", "resume2.png", "resume3.png"]
normalized_images, valid_indices = normalizer.image_normalizer.normalize_batch(image_paths)

print(f"Normalized {len(normalized_images)} valid images")
```

### 3. Normalization cho Extraction Data (Donut)

```python
# Chuẩn hóa dữ liệu extraction (ảnh + văn bản)
images = ["resume1.png", "resume2.png"]
texts = ["John Doe, Software Engineer", "Jane Smith, Data Scientist"]

normalized_images, normalized_texts, report = normalizer.normalize_extraction_data(
    images=images,
    texts=texts
)

print(f"Report:")
print(f"  Original samples: {report['original_samples']}")
print(f"  Normalized samples: {report['normalized_samples']}")
print(f"  Issues: {report['issues']}")
```

**Output:**
```
Report:
  Original samples: 2
  Normalized samples: 2
  Valid texts: 2
  Valid images: 2
  Issues: []
```

### 4. Normalization cho Ranking Data (BGE-M3)

```python
# Chuẩn hóa dữ liệu ranking (resume + JD + điểm)
resumes = [
    "John Doe - Python Developer with 5 years experience",
    "Jane Smith - Senior Data Scientist",
    "Invalid"  # Quá ngắn, sẽ bị lọc
]

job_descriptions = [
    "We need a Python developer with 5+ years experience",
    "Looking for senior data scientist",
    "Job description"
]

scores = [0.85, 0.92, 0.15]

normalized_resumes, normalized_jds, normalized_scores, labels, report = \
    normalizer.normalize_ranking_data(
        resumes=resumes,
        job_descriptions=job_descriptions,
        scores=scores
    )

print(f"Report:")
print(f"  Original samples: {report['original_samples']}")
print(f"  Normalized samples: {report['normalized_samples']}")
print(f"  Removed duplicates: Check issues")
print(f"  Removed outliers: Check issues")
```

## Configuration Options

### NormalizationConfig

```python
@dataclass
class NormalizationConfig:
    # Text normalization options
    lowercase: bool = True                      # Chuyển thành chữ thường
    remove_extra_whitespace: bool = True        # Xóa khoảng trắng thừa
    remove_special_chars: bool = False          # Xóa ký tự đặc biệt
    remove_numbers: bool = False                # Xóa số
    remove_punctuation: bool = False            # Xóa dấu câu
    normalize_unicode: bool = True              # Chuẩn hóa Unicode
    min_text_length: int = 10                   # Độ dài tối thiểu
    max_text_length: int = 10000                # Độ dài tối đa
    
    # Image normalization options
    target_image_size: Tuple[int, int] = (1280, 960)  # Kích thước target
    image_format: str = "RGB"                   # Format ảnh
    normalize_pixel_values: bool = True         # Chuẩn hóa pixel values
    apply_augmentation: bool = False            # Bật augmentation
    
    # Score normalization options
    normalize_scores: bool = True               # Chuẩn hóa điểm số
    score_min: float = 0.0                      # Điểm tối thiểu
    score_max: float = 1.0                      # Điểm tối đa
    
    # Data validation options
    remove_duplicates: bool = True              # Xóa bản sao
    handle_missing_values: bool = True          # Xử lý giá trị thiếu
    detect_outliers: bool = True                # Phát hiện outliers
    outlier_threshold: float = 3.0              # Z-score threshold
    
    # Data balancing options
    balance_data: bool = False                  # Cân bằng dữ liệu
    balance_strategy: str = "oversample"        # oversample hoặc undersample
```

## Advanced Usage

### 1. Custom Text Cleaning

```python
config = NormalizationConfig(
    lowercase=True,
    remove_special_chars=True,
    remove_numbers=True,
    remove_punctuation=False,  # Giữ lại dấu câu
)

normalizer = DataNormalizer(config)
```

### 2. Data Validation

```python
# Validate full dataset
texts = ["Text 1", "Text 2"]
values = [0.8, 0.9]
images = ["img1.png", "img2.png"]

report = normalizer.validator.validate_dataset(
    texts=texts,
    values=values,
    images=images
)

print(f"Total samples: {report['total_samples']}")
print(f"Valid texts: {report['valid_texts']}")
print(f"Valid values: {report['valid_values']}")
print(f"Valid images: {report['valid_images']}")
print(f"Issues: {report['issues']}")
```

### 3. Outlier Detection

```python
scores = [0.1, 0.5, 0.8, 0.9, 100.0]  # 100.0 is an outlier

filtered_scores, indices = normalizer.validator.detect_outliers(scores)

print(f"Original: {scores}")
print(f"Filtered: {filtered_scores}")
print(f"Removed outlier at index: {set(range(len(scores))) - set(indices)}")
```

### 4. Data Balancing

```python
# For imbalanced classification data
data = ["resume1", "resume2", "resume3", "resume4"]
labels = [1, 1, 1, 0]  # Class 1: 3 samples, Class 0: 1 sample

config = NormalizationConfig(
    balance_data=True,
    balance_strategy="oversample"  # Tăng lớp minority
)

normalizer = DataNormalizer(config)
balanced_data, balanced_labels = normalizer.balancer.balance(data, labels)

print(f"Original distribution: {Counter(labels)}")
print(f"Balanced distribution: {Counter(balanced_labels)}")
```

### 5. Image Augmentation

```python
config = NormalizationConfig(
    apply_augmentation=True,
    target_image_size=(1280, 960),
)

normalizer = DataNormalizer(config)

# Augmentation được áp dụng tự động trong normalize()
image = normalizer.image_normalizer.normalize("resume.png")

# Augmentation bao gồm:
# - Random rotation (±5 degrees)
# - Random brightness adjustment (0.9x - 1.1x)
# - Random contrast adjustment (0.9x - 1.1x)
```

## Integration với Pipeline

### Tự động normalization trong pipeline

```python
from cv_ranking_system.finetune_pipeline import (
    FineTuningOrchestrator, PipelineConfig
)

config = PipelineConfig(
    run_donut=True,
    run_bge=True,
    run_evaluation=True,
)

orchestrator = FineTuningOrchestrator(config)

# prepare_data() sẽ tự động:
# 1. Download datasets
# 2. Validate data integrity
# 3. Normalize texts, images, scores
# 4. Detect và remove outliers
# 5. Balance data nếu cần
# 6. Generate report

summary = orchestrator.run_pipeline()
```

### Manual normalization trước training

```python
from cv_ranking_system.data_normalizer import (
    DataNormalizer, NormalizationConfig
)
from cv_ranking_system.extraction.finetune_donut import (
    DonutFineTuner, DonutConfig
)

# Setup normalization
norm_config = NormalizationConfig(
    lowercase=True,
    remove_extra_whitespace=True,
    normalize_unicode=True,
    detect_outliers=True,
    balance_data=True,
)
normalizer = DataNormalizer(norm_config)

# Load raw data
raw_images = load_images("./data/images")
raw_texts = load_texts("./data/annotations.jsonl")

# Normalize
normalized_images, normalized_texts, report = \
    normalizer.normalize_extraction_data(raw_images, raw_texts)

# Setup training with normalized data
donut_config = DonutConfig(
    num_epochs=10,
    batch_size=8,
)

# Trong DonutFineTuner.prepare_datasets(), sử dụng normalized data
trainer = DonutFineTuner(donut_config)
trainer.train()  # Sẽ sử dụng dữ liệu đã chuẩn hóa
```

## Troubleshooting

### Problem: Quá nhiều dữ liệu bị lọc

**Solution:** Giảm các yêu cầu validation

```python
config = NormalizationConfig(
    min_text_length=5,          # Giảm từ 10
    max_text_length=50000,      # Tăng từ 10000
    detect_outliers=False,      # Tắt outlier detection
)
```

### Problem: Ảnh bị méo sau resize

**Solution:** Điều chỉnh augmentation hoặc target size

```python
config = NormalizationConfig(
    target_image_size=(1024, 768),  # Kích thước khác
    apply_augmentation=False,       # Tắt augmentation
)
```

### Problem: Mất quá nhiều dữ liệu khi balance

**Solution:** Sử dụng undersample thay vì oversample

```python
config = NormalizationConfig(
    balance_data=True,
    balance_strategy="undersample"  # Giảm lớp majority
)
```

## Performance Tips

1. **Enable Augmentation chỉ khi cần**
   - Augmentation làm chậm quá trình, bật chỉ khi data ít

2. **Outlier Detection có thể lọc dữ liệu hợp lệ**
   - Điều chỉnh `outlier_threshold` (mặc định: 3.0 std devs)

3. **Cân bằng dữ liệu có thể tăng size dataset**
   - Oversample làm tăng dataset size lên 2-3x
   - Undersample làm giảm dataset size

4. **Text Normalization costly operations**
   - Unicode normalization chậm với dataset lớn
   - Cân nhắc tắt nó nếu không cần thiết

## CLI Usage

```bash
# Normalize extraction data
python -m cv_ranking_system.data_normalizer normalize-extraction \
    --images-dir ./data/images \
    --annotations ./data/annotations.jsonl \
    --output ./normalized_data

# Normalize ranking data
python -m cv_ranking_system.data_normalizer normalize-ranking \
    --pairs-file ./data/resume_jd_pairs.jsonl \
    --output ./normalized_data

# Validate dataset
python -m cv_ranking_system.data_normalizer validate \
    --data-dir ./data \
    --report ./validation_report.json
```

## References

- [NFKD Unicode Normalization](https://en.wikipedia.org/wiki/Unicode_equivalence)
- [Z-score for Outlier Detection](https://en.wikipedia.org/wiki/Z-score)
- [Data Augmentation Best Practices](https://arxiv.org/abs/1809.02176)
- [Class Imbalance Handling](https://imbalanced-learn.org/)
