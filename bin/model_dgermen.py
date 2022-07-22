from tkinter import Variable
from turtle import hideturtle
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, TransformerConv, GINConv, PNAConv
from torch_geometric.nn import global_mean_pool
from torch.nn import BatchNorm1d
from torch.nn import ModuleList
import pytorch_lightning as pl

SEED = 42

class CustomGCN(torch.nn.Module):
    # TODO  Make the layers modular, can switch places of the layers with "init's" input
    # TODO  Can use mroe than one type of model
    def __init__(self, type, **kwargs):
        super(CustomGCN, self).__init__()
        pl.seed_everything(SEED)


        # Recording the parameters
        self.pars = kwargs
        
        self.dropout = self.pars["dropout"]
        # Extracting generic parameters
        self.num_node_features = self.pars["num_node_features"]
        self.gcn_hidden_neurons = self.pars["gcn_hidden_neurons"]

        # Extracting layer related parameters
        self.num_gcn_layers = self.pars["num_gcn_layers"]
        self.num_ff_layers = self.check_Key("num_ff_layers")
        self.ff_hidden_neurons = self.check_Key("ff_hidden_neurons")
        self.aggregators = self.check_Key("aggregators","list")
        self.scalers = self.check_Key("scalers","list")
        self.deg = self.check_Key("deg")

        # Setting generic model parameters
        model_pars_head = {
            "in_channels" : self.num_node_features,
            "out_channels"  : self.gcn_hidden_neurons
        }
        model_pars_rest = {
            "in_channels" : self.gcn_hidden_neurons,
            "out_channels"  : self.gcn_hidden_neurons
        }
        

        # Choosing the type of GCN
        try: 
            # WORKS
            if type == "GATConv":
                self.GCN_type_1 = GATConv

            # WORKS
            elif type == "TransformerConv":
                self.GCN_type_1 = TransformerConv

            # Currently not working, problem with model, nn = (int)
            # Need to feed a NN
            # PROBLEM Not working RN
            elif type == "GINConv":
                self.GCN_type_1 = GINConv

            # WORKS
            elif type == "PNAConv":
                self.GCN_type_1 = PNAConv
                model_pars_head["aggregators"] = self.aggregators
                model_pars_head["scalers"] = self.scalers
                model_pars_head["deg"] = self.deg
                
                model_pars_rest["aggregators"] = self.aggregators
                model_pars_rest["scalers"] = self.scalers
                model_pars_rest["deg"] = self.deg

            # No type name match case
            else:
                raise("Supplied model GCN type is not avaliable!")

        except KeyError:
            raise KeyError("Parameters are not proper")

        # Creating the layers

        

        # Module lists
        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        self.ff_layers = ModuleList()
        self.ff_batch_norms = ModuleList()

        # Initial GCN Layer -----
        self.convs.append(self.GCN_type_1(
            **model_pars_head,
        ))
        self.batch_norms.append(BatchNorm1d(self.gcn_hidden_neurons))

        # Other GCN Layers
        for _ in range(self.num_gcn_layers-1):
            self.convs.append(self.GCN_type_1(
                **model_pars_rest,
                ))
            self.batch_norms.append(BatchNorm1d(self.gcn_hidden_neurons))

        
        if self.num_ff_layers != 0:
            # Initial ff layer ----
            self.ff_layers.append(Linear(self.gcn_hidden_neurons, self.ff_hidden_neurons))
            self.ff_batch_norms.append(BatchNorm1d(self.ff_hidden_neurons))
            # Other ff layers
            for _ in range(self.num_ff_layers-2):
                self.ff_layers.append(Linear(self.ff_hidden_neurons, self.ff_hidden_neurons))
                self.ff_batch_norms.append(BatchNorm1d(self.ff_hidden_neurons))
            
            self.ff_layers.append(Linear(self.ff_hidden_neurons, 1))

    
    # Helper function to avoid getting keyError
    # If key doesnt exists default value is set to 0
    def check_Key(self, key ,expectedType = "int"):
        try:
            value = self.pars[key]
        except KeyError:
            if expectedType == "int":
                value = 0
            elif expectedType == "str":
                value = ""
            elif expectedType == "list":
                value = []
        return value


    def forward(self, x, edge_index, batch):

        # Conv layers
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, self.dropout, training=self.training)

        # Convulution Result Aggregation
        x = global_mean_pool(x, batch)  # [batch_size, gcn_hidden_neurons]

        # Classification according to ppoling
        for ff_l in self.ff_layers:
            x = F.relu(ff_l(x))
            # x = h + x  # residual#
            x = F.dropout(x, self.dropout, training=self.training)

        return x



