# STREAM: A Stochastic Gradient Relational Event Additive Model for modelling US patent citations from 1976 until 2022

This repository contains the codes for the paper [A Stochastic Gradient Relational Event Additive Model for modelling US patent citations from 1976 until 2022
](https://arxiv.org/abs/2303.07961), published in the *Journal of the Royal Statistical Society Series C*, 73(4), 2024.

## Abstract 

The patent citation network is a complex and dynamic system that reflects the diffusion of knowledge and innovation across different fields of technology. We focus on US patent citations from 1976 until 2022, which involves almost 10 million patents and over 100 million citations. Analyses of such networks are often limited to descriptive statistics. 
Instead, in this work we aim to develop a generative model for the citation process by combining relational event models (REMs) and machine learning techniques. We propose a stochastic gradient relational event additive model (STREAM) that models the relationships between cited and citing patents as events that occur over time, capturing the dynamic nature of the patent citation network. Each predictor in the generative model is assumed to have a non-linear behavior, modeled via a B-spline. By estimating the model through an efficient stochastic gradient descent approach, we are able to identify the key factors that drive the patent citation process. 
Our analysis revealed several interesting insights, such as the identification of time windows in which citations are more likely to happen, and the relevancy of the increasing number of citations received per patent. Overall, our results demonstrate the potential of the STREAM in capturing complex dynamics that arise in a large sparse network, maintaining the features and the interpretability, for which REMs are most famous.

## Repository structure

- `main.py`: entry point. The boolean flags at the top of the file select which stage to run (data preprocessing, model selection, model fitting).
- `data_preprocessing/`: cleaning of the USPTO grant and citation files (deduplication, removal of patents without an abstract), SBERT text embeddings of the abstracts, and construction of the event set (citing and cited patents joined with their publication dates, IPC classes and citation lags).
- `effects/`: construction and sampling of the risk set, and computation of the covariates (effects) used in the model.
- `model/`: B-spline basis, stochastic-gradient estimation of the relational event model, logit formulation, baseline and model selection.
- `utility.py`: shared helpers (deduplication, similarity measures).

## Data

The data were downloaded from the [USPTO bulk data repository](https://bulkdata.uspto.gov/) using the following [library](https://github.com/efm95/patents). The pipeline expects `grant_grant.csv`, `grant_cite.csv` and `grant_ipc.csv` in `data_preprocessing/`.

## How to run

1. Install the dependencies: `pandas`, `numpy`, `vaex`, `torch`, `sentence-transformers`, `tqdm`.
2. Compute the SBERT embeddings with `SBERT().fit()` from `data_preprocessing/text_embedding.py`, after the node list has been cleaned (a GPU is recommended; this call is commented out in `main.py`).
3. Set the flags at the top of `main.py` and run `python main.py`. Progress is logged to `rem.log`.

