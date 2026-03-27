from torch.utils.data import Dataset, random_split
from torch_geometric.data import DataLoader
from load_data import *
import os.path as osp

class GraphDataset(Dataset):
    def __init__(self, graph_list):
        self.graph_list = graph_list

    def __len__(self):
        return len(self.graph_list)

    def __getitem__(self, idx):
        return self.graph_list[idx]

class dataset():
    def __init__(self, dsType, labelType):
        super().__init__()
        self.dsDir = "/media/shulab/WD_10T/datasets/"
        self.dsType = dsType
        self.labelType = labelType

    def dspath_configuration(self, dsType):
        if dsType == "HCP":
            self.dsDir = osp.join(self.dsDir, "S1200")
            self.netDir = osp.join(self.dsDir, "network")
            self.netname = ["aal90_SC", "aal90_FC_merged"]
            self.labelfile = osp.join(self.dsDir, "S1200_889_CogScores.csv")
        elif dsType == "HCD":
            self.dsDir = osp.join(self.dsDir, "HCD")
            self.netDir = osp.join(self.dsDir, "network_csv")
            self.netname = ["bna246_SC", "bna246_FC"]
            self.labelfile = osp.join(self.dsDir, "HCD_471_CogScores.csv")

    def load_subj(self):
        with open(osp.join(self.dsDir, 'subjlist.txt'), "r") as f:
            slist = f.read().split('\n')
        return slist

    def setsubset(self, labelType, labeldim, split_ratio, create_val=False):
        self.dspath_configuration(self.dsType)
        self.subjlist = self.load_subj()
        self.datalist = load_data(netDir=self.netDir,
                                  subjlist=self.subjlist,
                                  netname=self.netname,
                                  labelfile=self.labelfile,
                                  labeltype=labelType,
                                  labeldim=labeldim,
                                  Isheader=True if self.dsType == "HCP" else False)
        
        train_size = int(split_ratio[0] * len(self.datalist))
        # print(train_size)
        if not create_val:
            test_size = len(self.datalist) - train_size
            self.train_dataset, self.test_dataset = random_split(self.datalist, [train_size, test_size])
            self.train_dataset, self.test_dataset = GraphDataset(self.train_dataset), GraphDataset(self.test_dataset)
        else:
            val_size = int(split_ratio[1] * len(self.datalist))
            test_size = len(self.datalist) - train_size - val_size
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.datalist, [train_size, val_size, test_size])


    def train_dataloader(self, batchsize):
        return DataLoader(self.train_dataset, batch_size=batchsize, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False)

