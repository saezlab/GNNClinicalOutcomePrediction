# Prediction of Clinical Features Using Graph Neural Networks

Discovery of functional motifs by association to clinical features using Graph Neural Networks. 

## Running 

### Training PNA For Regression

```bash
python train_test_controller.py --aggregators 'max' --bs 16 --dropout 0.0 --en my_experiment --epoch 200 --factor 0.8 --fcl 256 --gcn_h 64 --lr 0.001 --min_lr 0.0001 --model PNAConv --num_of_ff_layers 1 --num_of_gcn_layers 2 --patience 5 --scalers 'identity' --weight_decay 1e-05
```
### Training GAT For Regression
```bash
python train_test_controller.py --aggregators None --bs 16 --dropout 0.0 --en my_experiment --epoch 200 --factor 0.2 --fcl 128 --gcn_h 64 --lr 0.001 --min_lr 2e-05 --model GATConv --num_of_ff_layers 1 --num_of_gcn_layers 3 --patience 20 --scalers None --weight_decay 0
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



