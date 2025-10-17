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



## GR2ST pipeline

- Run `hvg_her2st.py` generation of highly variable genes.
- Run `data_precessing.ipynb` 
- Run `train_her2st.ipynb`
- Run `evel_her2st.py`

## Structure of GR2ST
<pre>
<code>
├── dataset
│   ├── her2st
│   │   └── data_precessing.ipynb
│   └── cscc
│       └── data_precessing.ipynb
└── GR2ST
    ├── dataset.py
    ├── model.py
    ├── utils.py
    ├── train_her2st.ipynb
    ├── train_cscc.ipynb
    ├── hvg_her2st.py
    ├── hvg_cscc.py
    ├── evel_her2st.py
    ├── evel_cscc.py  


</code>
</pre>
