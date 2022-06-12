from random import random
from sklearn.model_selection import KFold
import torch

# K-Fold cross validation index creator function
# Dataset idices and ratios must be supplied
# Return triplet of samplers for amount of wanted fold
def k_fold_TTV(dataset,T2VT_ratio,V2T_ratio):
    """Splits dataset into Train, Validation and Test sets

    Args:
        dataset (_type_): Data to be splitted
        T2VT_ratio (int): Train / (Valid + Test)
        V2T_ratio (int): Valid / Test
    """

    fold_T2VT=KFold(n_splits=T2VT_ratio+1)
    fold_V2T =KFold(n_splits=V2T_ratio+1)

    # List to save sampler triplet
    samplers = []

    # Fold count init
    fold_count = 0

    # Pulling indexes for sets
    for (train_idx, valid_test_idx) in fold_T2VT.split(dataset):
        for (valid_idx, test_idx) in fold_V2T.split(valid_test_idx):

            fold_count += 1

            valid_idx = valid_test_idx[valid_idx]
            test_idx = valid_test_idx[test_idx]


            samplers.append((
                (fold_count),
                (torch.utils.data.SubsetRandomSampler(train_idx)),
                (torch.utils.data.SubsetRandomSampler(test_idx)),
                (torch.utils.data.SubsetRandomSampler(valid_idx))))

    return samplers