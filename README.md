# ProtPhage: A Deep Learning Framework for Phage Viral Protein Identification and Functional Annotation

## 1. Introduction
This repository contains source data and code for paper "ProtPhage: A Deep Learning Framework for Phage Viral Protein Identification and Functional Annotation".
![](https://web-myitheima-tlias.oss-cn-beijing.aliyuncs.com/fig1.png)

## 2. Installation
```
python=3.8
pytorch=1.10.1
biopython=1.81
transformers=4.38.2
SentencePiece=0.2.0
scikit-learn=1.3.2
```
Notice:
1. You need install pretrained language modoel **ProtT5-XL-UniRef50**, the link is provided on [ProtT5-XL-U50](https://github.com/agemagician/ProtTrans#models).
2. You need to modify the model file path in the ``embedding.py`` of the project to the address where the ProtT5 model you downloaded is located.

## 3. Requirments
In order to run successfully, the embedding of ProtT5-XL-UniRef50 requires GPU. We utilized an NVIDIA GeForce RTX 3090 with 24GiB to embed peptide or protein sequences to 1024-dimensional vector.

Due to GitHub limitations, the ProtT5 embedding file "embedding.xlsx" are hosted on [Google Drive link](https://docs.google.com/spreadsheets/d/1KbiMomP5mkS54hZA9qQ1aBjCMzXB55U-/edit?usp=sharing&ouid=102664816674055186484&rtpof=true&sd=true)

## 4. Run
Run the files "predict_split_by_time.ipynb", "predict_split_by_smi.ipynb" and "predict_imbalance_data.ipynb"
