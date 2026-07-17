# TITAN-BBB

TITAN-BBB: Predicting BBB Permeability using Multi-Modal Deep-Learning Models

The paper is under review.

\[[Inference Model](https://huggingface.co/SaeedLab/TITAN-BBB)\] | \[[Dataset on HuggingFace](https://huggingface.co/datasets/SaeedLab/BBB)\] | \[[Cite](#citation)\]

## Abstract
Computational prediction of blood-brain barrier (BBB) permeability has emerged as a vital alternative to traditional experimental assays, which are often resource-intensive and low-throughput to meet the demands of early-stage drug discovery. While early machine learning approaches have shown promise, integration of traditional chemical descriptors with deep learning embeddings remains an underexplored frontier. In this paper, we introduce *TITAN-BBB*, a multi-modal deep-learning architecture that utilizes tabular, image, and text-based features and combines them using attention mechanisms. To evaluate, we aggregated multiple literature sources to create the largest BBB permeability dataset to date, enabling robust training for both classification and regression tasks. Our results demonstrate that TITAN-BBB achieves 86.5% of balanced accuracy on classification tasks and 0.436 of mean absolute error for regression, outperforming the state-of-the-art by 3.1 percentage points in balanced accuracy and reducing the regression error by 20%. Our approach also outperforms state-of-the-art models in both classification and regression performance, demonstrating the benefits of combining deep and domain-specific representations.

## System Requirements
- A computer with Ubuntu 16.04 (or later) or CentOS 8.1 (or later).
- CUDA-enabled GPU with at least 6 GB of memory.

## Installation Guide

### Install Anaconda
[Step by Step Guide to Install Anaconda](https://docs.anaconda.com/anaconda/install/)


### Fork the Repository
- Fork this repository to your own account.
- Clone your fork to your machine.

### Create a Conda Environment
```bash
cd <repository_directory>
conda env create --file environment.yml
```

### Activate the Environment
```bash
conda activate titan_bbb
```

## Running the Experiments
Below is an example command for running the paper experiments:

1. Setup to create the folders
```bash
cd src
python setup.py
```
---

2. Generate tabular features
```bash
cd src
python generate_tabular_features.py --task regression
python generate_tabular_features.py --task classification
```
---

3. Generate image embeddings
```bash
cd src
python generate_image_features.py --task regression
python generate_image_features.py --task classification
```
---

4. Generate text embeddings
```bash
cd src
python generate_text_features.py --task regression
python generate_text_features.py --task classification
```
---

5. Train and evaluate the model to classification and regression tasks
```bash
cd src
python train.py --task regression
python train.py --task classification
```
---

## Citation
The paper is under review. As soon as it is accepted, we will update this section.

## License

This model and associated code are released under the CC-BY-NC-ND 4.0 license and may only be used for non-commercial, academic research purposes with proper attribution. Any commercial use, sale, or other monetization of this model and its derivatives, which include models trained on outputs from the model or datasets created from the model, is prohibited and requires prior approval. Downloading the model requires prior registration on Hugging Face and agreeing to the terms of use. By downloading this model, you agree not to distribute, publish or reproduce a copy of the model. If another user within your organization wishes to use the model, they must register as an individual user and agree to comply with the terms of use. Users may not attempt to re-identify the deidentified data used to develop the underlying model. If you are a commercial entity, please contact the corresponding author.

## Contact

For any additional questions or comments, contact Fahad Saeed (fsaeed@fiu.edu).

