# To handle embeddings related jobs
import torch
from torch_geometric.loader import DataLoader

from torch_geometric.nn import GCNConv
from torch_geometric.utils import embedding

import config

seed = config.seed

# Extract embeddings for a given model and dataset
def get_intermediate_embeddings_for_dataset(model, dataset, batch_size=1, mode="FC", agg_method="mean"):
    intermediate_embeddings_list = []

    model.eval()  # Set the model to evaluation mode

    # Create a DataLoader for batching the dataset
    dataloader = DataLoader(dataset, batch_size=batch_size)

    related_data =[] 


    with torch.no_grad():
        for batch in dataloader:
            x = batch.x
            edge_index = batch.edge_index

            # Get intermediate embeddings using the provided function
            if mode == "FC":
                intermediate_embeddings = get_embeddings(model, x=x, edge_index=edge_index, batch=batch.batch)
            elif mode == "CNV":
                intermediate_embeddings = embedding.get_embeddings(model, x=x, edge_index=edge_index, batch=batch.batch)
                for indx, embed in enumerate(intermediate_embeddings):
                    if agg_method == "mean":
                        embed = torch.mean(embed, dim=0)
                        embed = embed.unsqueeze(0)
                    # Fill intermediate embeddings with the aggregated embeddings
                    intermediate_embeddings[indx] = embed

            # Append the intermediate embeddings to the list
            intermediate_embeddings_list.append(intermediate_embeddings)

            # Append the labels to the list
            related_data.append(batch)

    # Convert the list of intermediate embeddings into a PyTorch tensor
    layer_result_tensor_list = []
    for layer in range(len(intermediate_embeddings)):
        layer_result_list = [item[layer] for item in intermediate_embeddings_list]
        # Concatenate along the batch dimension
        layer_result_tensor = torch.cat(layer_result_list, dim=0)
        layer_result_tensor_list.append(layer_result_tensor)

    return layer_result_tensor_list, related_data


import warnings
from typing import Any, List

import torch
from torch import Tensor


def get_embeddings(
    model: torch.nn.Module,
    *args,
    **kwargs,
) -> List[Tensor]:
    """Returns the output embeddings of all
    :class:`~torch_geometric.nn.conv.MessagePassing` layers in
    :obj:`model`.

    Internally, this method registers forward hooks on all
    :class:`~torch_geometric.nn.conv.MessagePassing` layers of a :obj:`model`,
    and runs the forward pass of the :obj:`model` by calling
    :obj:`model(*args, **kwargs)`.

    Args:
        model (torch.nn.Module): The message passing model.
        *args: Arguments passed to the model.
        **kwargs (optional): Additional keyword arguments passed to the model.
    """
    from torch_geometric.nn import MessagePassing
    from torch.nn.modules.linear import Linear

    embeddings: List[Tensor] = []

    def hook(model: torch.nn.Module, inputs: Any, outputs: Any):
        # Clone output in case it will be later modified in-place:
        outputs = outputs[0] if isinstance(outputs, tuple) else outputs
        assert isinstance(outputs, Tensor)
        embeddings.append(outputs.clone())

    hook_handles = []
    for module in model.modules():  # Register forward hooks:
        if  isinstance(module, Linear):
            hook_handles.append(module.register_forward_hook(hook))

    if len(hook_handles) == 0:
        warnings.warn("The 'model' does not have any 'MessagePassing' layers")

    training = model.training
    model.eval()
    with torch.no_grad():
        model(*args, **kwargs)
    model.train(training)

    for handle in hook_handles:  # Remove hooks:
        handle.remove()

    return embeddings