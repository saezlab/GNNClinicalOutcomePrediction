import os
import torch
import pandas as pd
from torch_geometric.data import InMemoryDataset, download_url


class TissueDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["basel_zurich_preprocessed_compact_dataset.csv"]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def process(self):
        # Read data into huge `Data` list.
        self.data = pd.read_csv(os.path.join(self.root, "raw", self.raw_file_names[0]))
        grouped_by_image_num = self.data.groupby("ImageNumber").count()
        print(grouped_by_image_num)
        data_list = []

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        # torch.save((data, slices), self.processed_paths[0])


a = TissueDataset("../data")
print(a.raw_file_names)