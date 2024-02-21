import os
import custom_tools
from dataset import TissueDataset
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
# custom_tools.split_by_group()


dataset = TissueDataset(os.path.join(".", "../data/METABRIC", "month"),  "month")


count = 0
for data in dataset:
    if data.censored.item()==0:
        count+=1
    # print(data.censored.item())

print(count)
print(len(dataset))
import eval 

ci_index = eval.concordance_index([1,4,5,6], [0,0,0,0],[0,1,1,1])
print(ci_index)

# samplers = custom_tools.k_fold_ttv(dataset, 4, 1)


"""train_loader = DataLoader(self.dataset, batch_size=self.parser_args.bs, sampler= train_sampler)
validation_loader = DataLoader(self.dataset, batch_size=self.parser_args.bs, sampler= validation_sampler)
test_loader = DataLoader(self.dataset, batch_size=self.parser_args.bs, sampler= test_sampler)"""


"""
train_subset = Subset(dataset, samplers[0])
train_loader = DataLoader(train_subset, batch_size= samplers[0].size)


val_subset = Subset(dataset, samplers[1])
val_loader = DataLoader(val_subset, batch_size= samplers[1].size)

test_subset = Subset(dataset, samplers[2])
test_loader = DataLoader(test_subset, batch_size= samplers[2].size)

train_pids, val_pids, test_pids = None, None, None
for data in train_loader:
    print(len(set(data.p_id)), len(data.p_id))
    train_pids = set(data.p_id)


for data in val_loader:
    print(len(set(data.p_id)), len(data.p_id))
    val_pids = set(data.p_id)



for data in test_loader:
    print(len(set(data.p_id)), len(data.p_id))
    test_pids = set(data.p_id)

print(train_pids&val_pids)

print(train_pids)

print(val_pids)"""