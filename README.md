# Prediction of Clinical Features Using Graph Neural Networks

Discovery of functional motifs by association to clinical features using Graph Neural Networks. 


## GNNExplainer
In order to run the ```GNNExplainer```, go to ```bin``` and run the following!

```bash
python gnnexplainer.py --aggregators 'max' --bs 16 --dropout 0.0 --fcl 256 --gcn_h 64 --model PNAConv --num_of_ff_layers 1 --num_of_gcn_layers 2 --scalers 'identity' --idx 20
```
