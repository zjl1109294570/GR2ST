# GR2ST: Spatial Transcriptomics Prediction based on Graph-Enhanced Multimodal Contrastive Learning

## Overview
Here, we propose GR2ST, a deep learning model that learns the underlying connections between image features and gene expression to predict spatial transcriptomics. GR2ST employs a large pre-trained model for the purpose of image feature extraction; a dual-branch graph neural network captures functional similarity and spatial proximity in gene expression by incorporating spatial coordinates and cell-type annotations. Contrastive learning then narrows the discrepancy separating image features from gene expression representations. Finally, we use the image features aligned by GR2ST to predict gene expression. We evaluated the performance of the model on datasets from cutaneous squamous cell carcinoma and human breast cancer to demonstrate its effectiveness.

![(Variational)](GR2ST/GR2ST/model.png)

## System environment
Required package:
- PyTorch >= 2.1.0
- scanpy >= 1.8
- python >=3.9

## Datasets
Two publicly available ST datasets were used in this study. You can find them on the following websites：
-  human HER2-positive breast tumor ST data from https://github.com/almaan/her2st/.
-  human cutaneous squamous cell carcinoma 10x Visium data from GSE144240.

## Baseline
You can obtain the baseline used in this article from the following link.
- HisToGene: https://github.com/maxpmx/HisToGene
- Hist2ST: https://github.com/biomed-AI/Hist2ST
- THItoGene: https://github.com/yrjia1015/THItoGene
- HGGEP: https://github.com/QSong-github/HGGEP
- mclSTExp: https://github.com/shizhiceng/mclSTExp
- Reg2ST: https://github.com/Holly-Wang/Reg2ST

## GR2ST pipeline

- Run `hvg_her2st.py` generation of highly variable genes.
- Run `data_precessing.ipynb` to obtain the various data required for model training.
- Run `train_her2st.ipynb` to train the model on the her2st dataset using leave-one-out cross-validation.
- Run `evel_her2st.py` to calculate the PCC between predicted and ground truth gene expression to evaluate model performance.

## Results
![her2st result](GR2ST/GR2ST/her2st_output.png)

![cscc result](GR2ST/GR2ST/cscc_output.png)

| Model       | PCC (HER2+) | PCC (cSCC) | 
|-------------|------------------|------------------|
| HisToGene   | 0.0818           | 0.0771           |
| Hist2ST     | 0.1484           | 0.1749           |
| THIToGene   | 0.1330           | 0.1796           |
| mclSTExp    | 0.2281           | 0.3157           |
| HGGEP       | 0.1566           | 0.1084           |
| Reg2ST      | 0.1741           | 0.2024      |
| GR2ST       | **0.2340**           |  **0.3246**      |

Reuslts of ablation study and parameter sensitivity are in folder `results`.
</code>
</pre>
