# Resume NER Training Dataset

## Overview
This dataset contains 5,960 standardized resume samples for Named Entity Recognition (NER) training, combining four high-quality resume datasets:

- **Kaggle ATS Dataset**: 220 samples - Resume scoring with entity annotations
- **HuggingFace NER Dataset**: 4,971 samples - PDF resumes with skill annotations  
- **Resume Corpus Dataset**: 224 samples - 36 entity types across diverse resumes
- **Doccano Dataset**: 545 samples - Professional CVs in Doccano format

## Data Format
Each sample contains:
```json
{
  "text": "Full resume text content...",
  "annotations": [
    [start_position, end_position, "ENTITY_LABEL"],
    [start_position, end_position, "ENTITY_LABEL"]
  ]
}
```

## Entity Labels (14 categories)
- **SKILL**: Technical skills, tools, and competencies (549,465 instances)
- **DESIGNATION**: Job titles and positions (4,301 instances)
- **LOCATION**: Cities, states, and geographical locations (4,073 instances)
- **EXPERIENCE**: Work experience and duration (3,544 instances)
- **PERSON**: Names and personal information (3,122 instances)
- **EDUCATION**: Degrees, colleges, and educational background (2,124 instances)
- **EXPERTISE**: Areas of professional expertise (1,045 instances)
- **EMAIL**: Contact email addresses (815 instances)
- **COMPANY**: Company and organization names (218 instances)
- **COLLABORATION**: Teamwork and collaboration skills (187 instances)
- **LANGUAGE**: Language proficiencies (159 instances)
- **ACTION**: Professional actions and responsibilities (133 instances)
- **CERTIFICATION**: Professional certifications (122 instances)
- **OTHER**: Miscellaneous entities (10,267 instances)

## Usage
Perfect for training NER models on resume data. All annotations are:
- ✅ Standardized to consistent format
- ✅ Validated for text alignment
- ✅ Ready for immediate training use
- ✅ Compatible with HuggingFace transformers

## Source Datasets
- [Kaggle ATS Dataset](https://www.kaggle.com/datasets/mgmitesh/ats-scoring-dataset)
- [HuggingFace NER Dataset](https://huggingface.co/datasets/Mehyaar/Annotated_NER_PDF_Resumes)
- [Resume Corpus Dataset](https://github.com/vrundag91/Resume-Corpus-Dataset)
- [Doccano Dataset](https://github.com/juanfpinzon/resume-dataset)

## License
MIT License - Use freely for research and commercial purposes.
