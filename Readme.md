

# DIHGNN

This repository is an implementation of **Enhancing Smart Contract Vulnerability Detection via Expression-Level Control Flow Graph and Infinite-Hop Graph Neural Network**

Due to the storage limitations of the anonymous repository, we are unable to upload the processed dataset files to this repository. After the paper is accepted, we will make the full dataset available on GitHub.

Below is the file directory introduction of this repository:

1. `bert-tiny`: An open-source model containing bert-tiny.
2. `dataset`: Contains contract source code from the SmartBugs dataset, as well as the Code Example provided in our paper.
3. `Step1_CFGgeneration`
   1. `SCFG_ECFG.py`: Generates nodes and edges for SCFG and ECFG, and provides visualization.
4. `Step2_NodeEmbedding`
   1. `normalize_CFG.py`: Extracts node attributes for SCFG and ECFG.
   2. `Embedding.py`: Embeds nodes for SCFG and ECFG.
5. `Step3_Classification`
   1. `config.py`: Configuration file for models, datasets, and hyperparameters.
   2. `main.py`: Code for model training and testing.
   3. `model.py`: Model including DIHGNN, GAT, and GCN.
   4. `result.py`: Visualizes model training results, including F1 scores, other metrics, and loss curves.
   5. `Log`: Model training results. You can use `result.py` to view the training results of each model in `Log`.











