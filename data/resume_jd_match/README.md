---
dataset_info:
  features:
  - name: text
    dtype: string
  - name: label
    dtype: string
  splits:
  - name: train
    num_bytes: 53916295
    num_examples: 6241
  - name: test
    num_bytes: 15298281
    num_examples: 1759
  download_size: 37077813
  dataset_size: 69214576
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
---
