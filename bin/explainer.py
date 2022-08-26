

class Explainer:
 
    # init method or constructor
    def __init__(self, model, dataloader=None):
        self.model = model
        self.dataloader = dataloader
 
    # Sample Method
    def say_hi(self):
        print('Hello, my name is', self.name)

    def set_dataloader(self, dataloader):
        self.dataloader = dataloader

    def explain_by_gnnexplainer(self, epoch, return_type, feat_mask_type):
        explainer = pyg.nn.GNNExplainer(self.model, epochs = args.exp_epoch, lr = args.exp_lr, 
                                    return_type = args.retun_type, feat_mask_type = args.feat_mask_type).to(device)
    
        result = explainer.explain_graph(test_graph.x.to(device), test_graph.edge_index.to(device))
        
        (feature_mask, edge_mask) = result
        edges_idx = edge_mask > args.relevant_edges
        explanation = pyg.data.Data(test_graph.x, test_graph.edge_index[:, edges_idx])

        explanation = pyg.transforms.RemoveIsolatedNodes()(pyg.transforms.ToUndirected()(explanation))
    
        return explanation
 
 
p = Person('Nikhil')
p.say_hi()
