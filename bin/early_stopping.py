import numpy as np
import torch
import custom_tools

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=3, verbose=False, delta=0, model_path='some_path', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = model_path
        self.trace_func = trace_func
        self.best_eval_score=None
        # self.best_model = None
    def __call__(self, val_loss, eval_score, model, hyperparams, id_file_name, deg=None):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_eval_score = eval_score
            self.save_model(val_loss,eval_score, model, hyperparams, id_file_name, deg)
        elif score < self.best_score + self.delta:
            self.counter += 1
            # self.trace_func(f'Patience: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            
            self.best_score = score
            if eval_score > self.best_eval_score:
                self.best_eval_score = eval_score
                self.save_model(val_loss, eval_score, model, hyperparams, id_file_name, deg)
            self.counter = 0
    
    # (model: CustomGCN,fileName ,mode: str, path = os.path.join(os.curdir, "..", "models")
    def save_model(self, val_loss, eval_score, model, hyperparams, id_file_name, deg):
        '''Saves model when validation loss decrease.'''
        
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.2f} --> {val_loss:.2f}).  Saving checkpoint model ...')
            self.trace_func(f'Best eval score increased ({self.best_eval_score:.2f} --> {eval_score:.2f}).  Saving checkpoint model ...')
        self.val_loss_min = val_loss
        self.best_eval_score = eval_score

        # vars(self.parser_args)

        custom_tools.save_dict_as_json(hyperparams, id_file_name, self.path)
        custom_tools.save_model(model=model, fileName=id_file_name, mode="SD", path=self.path)
        if deg!=None:
            
            custom_tools.save_pickle(deg, f"{id_file_name}_deg.pckl", self.path)
        