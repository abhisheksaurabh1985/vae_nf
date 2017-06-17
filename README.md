# Variational Autoencoder with Normalizing Flows

**REFERENCES**
1. Auto-Encoding Variational Bayes: https://arxiv.org/abs/1312.6114
2. Variational Inference with Normalizing Flows: https://arxiv.org/abs/1505.05770

**HOW TO RUN**
*norm_flow.py* should be executed. Parameters defined in *config.py*.  

**TO DO**
1. Output directory structure is hard-coded in *config.py*. To be automated. 
2. In case of planar normalizing flow, cost becomes NaNs for higher values of flows (typically greater than 8). I will resolve this issue at the earliest possible instance. Though it is not very high on priority. 

