# STREAM: A Stochastic Gradient Relational Event Additive Model for modelling US patent citations from 1976 until 2022

This repository contains the codes for the paper [A Stochastic Gradient Relational Event Additive Model for modelling US patent citations from 1976 until 2022
](https://arxiv.org/abs/2303.07961).

## Abstract 

The patent citation network is a complex and dynamic system that reflects the diffusion of knowledge and innovation across different fields of technology. We focus on US patent citations from 1976 until 2022, which involves almost 10 million patents and over 100 million citations. Analyses of such networks are often limited to descriptive statistics. 
Instead, in this work we aim to develop a generative model for the citation process by combining relational event models (REMs) and machine learning techniques. We propose a stochastic gradient relational event additive model (STREAM) that models the relationships between cited and citing patents as events that occur over time, capturing the dynamic nature of the patent citation network. Each predictor in the generative model is assumed to have a non-linear behavior, modeled via a B-spline. By estimating the model through an efficient stochastic gradient descent approach, we are able to identify the key factors that drive the patent citation process. 
Our analysis revealed several interesting insights, such as the identification of time windows in which citations are more likely to happen, and the relevancy of the increasing number of citations received per patent. Overall, our results demonstrate the potential of the STREAM in capturing complex dynamics that arise in a large sparse network, maintaining the features and the interpretability, for which REMs are most famous.

