# Prediction of Clinical Features Using Graph Neural Networks

Discovery of functional motifs by association to clinical features using Graph Neural Networks. 

## Running 

### Training PNA For Regression

```bash
python train_test_dgermen.py --aggregators 'max' --bs 16 --dropout 0.0 --en my_experiment --epoch 200 --factor 0.8 --fcl 256 --gcn_h 64 --lr 0.001 --min_lr 0.0001 --model PNAConv --num_of_ff_layers 1 --num_of_gcn_layers 2 --patience 5 --scalers 'identity' --weight_decay 1e-05
```

### GNNExplainer For PNA Regressor
```bash
python gnnexplainer.py --aggregators 'max' --bs 16 --dropout 0.0 --fcl 256 --gcn_h 64 --model PNAConv --num_of_ff_layers 1 --num_of_gcn_layers 2 --scalers 'identity' --idx 10
```
### LIME Explainer for PNA Regressor
```bash
python lime.py --aggregators 'max' --bs 16 --dropout 0.0 --fcl 256 --gcn_h 64 --model PNAConv --num_of_ff_layers 1 --num_of_gcn_layers 2 --scalers 'identity' --idx 10
```

### SHAP Explainer for PNA Regressor
```bash
python shap.py --aggregators 'max' --bs 16 --dropout 0.0 --fcl 256 --gcn_h 64 --model PNAConv --num_of_ff_layers 1 --num_of_gcn_layers 2 --scalers 'identity' --idx 10
```

## Explainable cells and cell interactions


| Original Graph                                                                                              | SubGraph                                                                                               |
|------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| ![Original Graph](https://github.com/saezlab/GNNClinicalOutcomePrediction/blob/main/plots/original_graphs/original_graph_28_50_0.001_regression_individual_feature.png) | ![QualitativeResults](https://github.com/saezlab/GNNClinicalOutcomePrediction/blob/main/plots/subgraphs/subgraph_28_50_0.001_regression_individual_feature.png) |

### First Results of Hyperparameter Tuning

![Explainer Results](https://github.com/saezlab/GNNClinicalOutcomePrediction/blob/main/plots/subgraphs/futon_explainer.gif)

## Resources

* [Awesome graph explainability papers](https://github.com/flyingdoog/awesome-graph-explainability-papers)
* [Towards Explainable Graph Neural Networks](https://towardsdatascience.com/towards-explainable-graph-neural-networks-45f5e3912dd0)
* [Parameterized Explainer for Graph Neural Networks (Github-PyTorch)](https://github.com/LarsHoldijk/RE-ParameterizedExplainerForGraphNeuralNetworks)
* [Parameterized Explainer for Graph Neural Networks (Github-TensorFlow)](https://github.com/flyingdoog/PGExplainer)
* [GNNExplainer - DGL](https://docs.dgl.ai/en/0.8.x/generated/dgl.nn.pytorch.explain.GNNExplainer.html)
* [GNNExplainer on MUTAG (Graph Classification) - Colab](https://colab.research.google.com/drive/14GPEIR7uRz50K9E_p9OUwjSOig0ZOB_E?usp=sharing)
* [GNNExplainer on MUTAG (Graph Classification) - Colab 2](https://colab.research.google.com/drive/1fLJbFPz0yMCQg81DdCP5I8jXw9LoggKO?usp=sharing#scrollTo=g35SSQ3oukNM)
* [Captum - Paper](https://arxiv.org/pdf/2009.07896.pdf)
* [Captum - Website](https://captum.ai/)
* [Explainability in Graph Neural Networks](https://mars-tin.github.io/archives/reading/gnn_explain/)
* [Distributed Computation of Attributions using Captum](https://github.com/pytorch/captum/blob/master/tutorials/Distributed_Attribution.ipynb)
* [Workshop on GNNExplainer with Graph Classification (MUTAG)](https://colab.research.google.com/github/VisiumCH/AMLD-2021-Graphs/blob/master/notebooks/workshop_notebook.ipynb#scrollTo=aSnkQfG4gnsc)
* [Extending Gnnexplainer for graph classification - PyG](https://github.com/pyg-team/pytorch_geometric/pull/2597)
----------
* [SHAP-Library](https://github.com/slundberg/shap)
* [GraphSVX: Shapley Value Explanations for Graph Neural Networks](https://arxiv.org/abs/2104.10482)
* [The Shapley Value in Machine Learning](https://arxiv.org/abs/2202.05594)
* [GRAPHSHAP: Motif-based Explanations for Black-box Graph Classifiers](https://arxiv.org/abs/2202.08815)
* [GraphLIME: Local Interpretable Model Explanations for Graph Neural Networks](https://ieeexplore.ieee.org/abstract/document/9811416?casa_token=LKVeyUFi1BEAAAAA:JXLkxY4qYRKiF-06Uh4tFz-Bsj_w_Do17CJLTq1afciKtHkq42Snkg-ttkaySm5LkAQmrI61rx1R)
* [CF-GNNExplainer: Counterfactual Explanations for Graph Neural Networks](https://proceedings.mlr.press/v151/lucic22a.html)
* [Explaining Graph Neural Networks with Structure-Aware Cooperative Games](https://arxiv.org/abs/2201.12380)
* [Reliable Graph Neural Network Explanations Through Adversarial Training](https://arxiv.org/abs/2106.13427)
* [Explainable AI Video Series](https://www.youtube.com/watch?v=OZJ1IgSgP9E&list=PLV8yxwGOxvvovp-j6ztxhF3QcKXT6vORU)
* [awesome-machine-learning-interpretability](https://github.com/jphall663/awesome-machine-learning-interpretability)
* [PGM-Explainer: Probabilistic Graphical Model Explanations for Graph Neural Networks](https://par.nsf.gov/servlets/purl/10200285)
* [An Explainable AI Library for Benchmarking Graph Explainers-GXAI](https://graph-learning-benchmarks.github.io/assets/papers/glb2022/An_Explainable_AI_Library_for_Benchmarking_Graph_Explainers.pdf) [[Code](https://github.com/mims-harvard/GXAI-Bench)]
* [On Explainability of Graph Neural Networks via Subgraph Explorations](https://arxiv.org/abs/2102.05152) [[Code](https://github.com/divelab/DIG/tree/main/dig/xgraph/SubgraphX)]
* [Data Representing Ground-Truth Explanations to Evaluate XAI Methods](https://arxiv.org/pdf/2011.09892.pdf)
* [Towards Ground Truth Explainability on Tabular Data](https://arxiv.org/pdf/2007.10532.pdf)
* [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/index.html)
* [“Why Should I Trust You?” Explaining the Predictions of Any Classifier](https://arxiv.org/pdf/1602.04938.pdf)
* [Quantus](https://github.com/understandable-machine-intelligence-lab/Quantus)
* [Papers and code of Explainable AI esp. w.r.t. Image classificiation](https://github.com/samzabdiel/XAI)
* [OpenXAI](https://github.com/AI4LIFE-GROUP/OpenXAI)
* [How can I choose an explainer? An Application-grounded Evaluation of Post-hoc Explanations](https://arxiv.org/abs/2101.08758)


Biological/Biomedicine Papers & Repos That used/cited GNNExplainer Paper
----------
* [histocartography](https://github.com/histocartography/histocartography)
* [Towards Explainable Graph Representations in Digital Pathology](https://arxiv.org/pdf/2007.00311.pdf)
* [An Causal XAI Diagnostic Model for Breast Cancer Based on Mammography Reports](https://ieeexplore.ieee.org/abstract/document/9669648?casa_token=n5V-TdpnRC4AAAAA:ykLyVwcFQ1EtF0A4ihXMxcisKndreyb7xdlXtiu5UMPA_RYUlFmBtMDEz03n98r_2Lf6hFV01dK2)
* [Predicting Cell Type and Extracting Key Genes using Single Cell Multi-Omics Data and Graph Neural Networks](https://cs.brown.edu/research/pubs/theses/ugrad/2022/zaki.hossam.pdf)
* [Graph Representation Learning in Biomedicine](https://arxiv.org/pdf/2104.04883.pdf)
* [Predicting the Survival of Cancer Patients With Multimodal Graph Neural Network](https://ieeexplore.ieee.org/abstract/document/9440752?casa_token=xZ71hc0rjT0AAAAA:wl7e1yWd3G7N3SQv6rNji2b6z1jy86yCjbdjLMiSDL2TuOyXZE6GezHz-z94RCBVWxgfhOJRjY4F)
* [scDeepSort: a pre-trained cell-type annotation method for single-cell transcriptomics using deep learning with a weighted graph neural network](https://academic.oup.com/nar/article/49/21/e122/6368052?login=true) [[Code](https://github.com/ZJUFanLab/scDeepSort)]
* [A survey on graph-based deep learning for computational histopathology](https://www.sciencedirect.com/science/article/pii/S0895611121001762)
* [Explaining decisions of graph convolutional neural networks: patient-specific molecular subnetworks responsible for metastasis prediction in breast cancer](https://link.springer.com/article/10.1186/s13073-021-00845-7)


