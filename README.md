# Pneumonia Classification with ResNet18: Effects of Preprocessing and Transfer Learning

## Overview
This repository contains the code, results, and figures for a research study
examining the separate and combined effects of image preprocessing and
transfer learning on the performance of a ResNet18 deep learning model in
classifying chest X-ray images as normal or pneumonia-affected.

## Research Design
A 2×2 factorial experimental design was used, producing four configurations:

| Configuration | Model Initialization | Preprocessing |
|---|---|---|
| A | Training from scratch | Basic |
| B | Training from scratch | Additional |
| C | Transfer learning | Basic |
| D | Transfer learning | Additional |

## Dataset
- **Source:** Chest X-Ray Images (Pneumonia), Kermany et al., available via
  Mendeley Data: https://data.mendeley.com/datasets/rscbjbr9sj/2
- **Total images:** 5,856 (1,583 NORMAL / 4,273 PNEUMONIA)
- **Split:** 70% training / 15% validation / 15% test (stratified, fixed seed = 42)
- The dataset itself is **not included** in this repository due to size and
  licensing; use the link above to download it.

## Methodology Summary
- **Model:** ResNet18 (PyTorch/Torchvision), final layer replaced for binary
  classification.
- **Training:** 10 epochs, batch size 32, learning rate 0.0001, Adam
  optimizer, CrossEntropyLoss.
- **Environment:** Google Colab (GPU: Tesla T4).
- **Evaluation metrics:** Accuracy, Precision, Sensitivity, Specificity,
  F1-score, AUROC, and confusion matrices.

## Results Summary
Transfer learning was the dominant factor in improving performance:

| Config | Accuracy | AUROC |
|---|---|---|
| A | 0.9704 | 0.9896 |
| B | 0.9556 | 0.9876 |
| C | 0.9841 | 0.9974 |
| D | 0.9852 | 0.9974 |

Configuration D (transfer learning + additional preprocessing) achieved the
best overall performance. Additional preprocessing alone showed a limited,
inconsistent effect depending on whether transfer learning was applied.

## Repository Contents
- `notebook.ipynb` — full training and evaluation code
- `comparison_table.csv` — full test metrics for all four configurations
- `split_summary.csv` — dataset split summary
- `confusion_matrices.csv` — TP/FN/TN/FP values per configuration
- `figures/` — class distribution, training curves, metrics comparison,
  and confusion matrices (combined heatmap)

## Requirements
Python 3.x
PyTorch
Torchvision
scikit-learn
pandas
numpy
matplotlib

## Disclaimer
This model performs a computational classification task and is not a
substitute for professional medical diagnosis. It has not been validated on
external data from other institutions or imaging devices.

## Author
Amina Abdualjalil Ali Muzammil
Islamic University of Minnesota — Bachelor's Graduation Research
