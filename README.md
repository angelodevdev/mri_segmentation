# MRI Brain Tumor Classification & Segmentation

## Overview
This repository contains a MATLAB pipeline for the analysis of Magnetic Resonance Imaging (MRI) scans. The project explores two distinct machine learning paradigms:
1. **Supervised Learning (SVM):** To classify MRI scans as either "Healthy" or "Pathological".
2. **Unsupervised Learning (PCA & Spectral Clustering):** To segment individual images and isolate potential anomalous regions (e.g., tumors) based on pixel intensity and spatial relationships.

## Baseline Results & Discussion (SVM)
The classification model was evaluated on a held-out test set (30% of the data). Since this is a baseline model trained on **raw flattened pixels** (without advanced feature extraction), the results reflect the limitations of not preserving spatial hierarchies:

* **Sensitivity (True Positive Rate): ~80.4%** The model is quite effective at identifying actual pathological cases, which is crucial in a medical context to minimize False Negatives.
* **Precision: ~68.5%**
  When the model predicts a pathology, it is correct about 68.5% of the time.
* **Specificity (True Negative Rate): ~37.0%**
  The model struggles to accurately identify healthy brains, resulting in a high rate of False Positives. This is expected for linear classifiers on complex imaging data without spatial context.
  
## Dataset Structure
To run the code, you need to provide your own dataset or download a public one. Organise the images in the root directory following this exact structure:
```text
/dataset
    /no   # Contains images of healthy brains (Label: 0)
    /yes  # Contains images with tumors/pathologies (Label: 1)
