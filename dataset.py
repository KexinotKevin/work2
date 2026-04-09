from torch.utils.data import Dataset, random_split, Subset
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold
import numpy as np
from load_data import *
import os.path as osp
import pandas as pd
from datasets_cfg import get_dataset_cfg


SC_KIND = {
    "fiber_length": "connectome_mean_length_10M.csv",
    "FA": "connectome_mean_FA_10M.csv",
    "fiber_bundle_capacity": "connectome_sift2_fbc_10M.csv",
    "fiber_count": "connectome_streamline_count_10M.csv",
}

FC_KIND = {
    "pcc_rest": "pFC.csv",
}


def resolve_kind_filename(kind_map, value):
    value = str(value).strip()
    if not value:
        raise ValueError("kind value must be non-empty.")
    return kind_map.get(value, value)


def parse_sc_kinds(value):
    if isinstance(value, (list, tuple)):
        items = [str(v).strip() for v in value if str(v).strip()]
    else:
        items = [x.strip() for x in str(value).split(",") if x.strip()]
    if not items:
        raise ValueError("sc_kind/sc_kinds must be non-empty.")
    return items

class GraphDataset(Dataset):
    def __init__(self, graph_list):
        self.graph_list = graph_list

    def __len__(self):
        return len(self.graph_list)

    def __getitem__(self, idx):
        return self.graph_list[idx]

class dataset():
    def __init__(
        self,
        dsType,
        labelType,
        *,
        use_dataset_cfg=False,
        dataset_name=None,
        atlas_name="bna246",
        sc_kind="fiber_count",
        sc_kinds=None,
        fc_kind="pcc_rest",
        output_dir=None,
    ):
        super().__init__()
        self.dsDir = "/public/home/baitianyu/kexin/datasets/"
        self.dsType = dsType
        self.labelType = labelType
        self.use_dataset_cfg = use_dataset_cfg
        self.dataset_name = dataset_name or dsType
        self.atlas_name = atlas_name
        self.sc_kind = sc_kind
        self.sc_kinds = parse_sc_kinds(sc_kinds if sc_kinds is not None else sc_kind)
        self.fc_kind = fc_kind
        self.subject_col = "Subject"
        self.Isheader = False
        self.use_cfg_layout = False
        self.output_dir = output_dir
        self.use_mat_format = False
        self.matDir = None

    def dspath_configuration(self, dsType):
        if self.use_dataset_cfg:
            dt_cfg = get_dataset_cfg(self.dataset_name)
            self.dsDir = dt_cfg.get("conn_dir", "")
            self.netDir = self.dsDir
            self.labelfile = dt_cfg.get("scores_path", "")
            tgt_cols = dt_cfg.get("tgt_label_list", [])
            if len(tgt_cols) < 4:
                raise ValueError(f"dataset cfg invalid for {self.dataset_name}.")
            self.subject_col = tgt_cols[0]
            # 提取性别和年龄列（用于 GRL 对抗训练）
            self.gender_col = tgt_cols[1] if len(tgt_cols) > 1 else None
            self.age_col = tgt_cols[2] if len(tgt_cols) > 2 else None
            self.sc_netnames = [
                resolve_kind_filename(SC_KIND, item) for item in self.sc_kinds
            ]
            self.fc_netname = resolve_kind_filename(FC_KIND, self.fc_kind)
            # keep old attr for compatibility
            self.netname = self.sc_netnames + [self.fc_netname]
            self.use_cfg_layout = True
            self.Isheader = True
            # ====== 【HCD使用.mat格式】 ======
            self.use_mat_format = (self.dataset_name.upper() == "HCD")
            self.matDir = self.dsDir
            return

        if dsType == "HCP":
            self.dsDir = osp.join(self.dsDir, "S1200")
            self.netDir = osp.join(self.dsDir, "network")
            self.netname = ["aal90_SC", "aal90_FC_merged"]
            self.labelfile = osp.join(self.dsDir, "S1200_889_CogScores.csv")
            self.subject_col = "Subject"
            self.use_cfg_layout = False
            self.Isheader = True
            self.sc_netnames = [self.netname[0]]
            self.fc_netname = self.netname[1]
        elif dsType == "HCD":
            self.dsDir = osp.join(self.dsDir, "HCD")
            self.netDir = osp.join(self.dsDir, "network_mat")
            self.matDir = osp.join(self.dsDir, "network_mat")
            self.netname = ["bna246_SC", "bna246_FC"]
            self.labelfile = osp.join(self.dsDir, "HCD_471_CogScores.csv")
            self.subject_col = "Subject"
            self.use_cfg_layout = False
            self.Isheader = False
            self.use_mat_format = True
            self.sc_netnames = [self.netname[0]]
            self.fc_netname = self.netname[1]

    def load_subj(self):
        if self.use_cfg_layout:
            dt = pd.read_csv(self.labelfile)
            if self.subject_col not in dt.columns:
                raise KeyError(f"subject column `{self.subject_col}` not found in {self.labelfile}")
            return dt[self.subject_col].astype(str).tolist()
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
                                  Isheader=self.Isheader,
                                  sc_netnames=self.sc_netnames,
                                  fc_netname=self.fc_netname,
                                  subject_col=self.subject_col,
                                  use_cfg_layout=self.use_cfg_layout,
                                  atlas_name=self.atlas_name,
                                  gender_col=self.gender_col,
                                  age_col=self.age_col,
                                  output_dir=self.output_dir,
                                  matDir=self.matDir,
                                  use_mat_format=self.use_mat_format)
        
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
        # drop_last：避免最后一个 batch 仅含 1 张图时 BatchNorm（如 BrainGNN 的 bn1/bn2）在 train 模式下报错
        return DataLoader(
            self.train_dataset, batch_size=batchsize, shuffle=True, drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False)

    def get_k_fold_splits(self, n_splits=10, shuffle=True, random_state=None):
        """生成 K 折交叉验证的索引划分。

        Args:
            n_splits: 折数，默认 10
            shuffle: 是否在划分前打乱数据
            random_state: 随机种子，确保可复现

        Returns:
            kfold: KFold 对象，可用于生成训练/验证/测试集索引
            kfold_indices: 包含 n_splits 个元素的列表，每个元素为
                          (train_idx, val_idx, test_idx) 元组
        """
        kfold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        return kfold

    def create_k_fold_dataloaders(self, kfold, fold, batchsize, labeldim=64, split_ratio=None, random_state=None):
        """根据 KFold 划分创建特定 fold 的数据加载器。

        Args:
            kfold: KFold 对象
            fold: 当前折的索引 (0 到 n_splits-1)
            batchsize: 批大小
            labeldim: 标签维度（保留与 setsubset 一致，当前未使用）
            split_ratio: 训练/验证/测试比例列表，如 [0.8, 0.1, 0.1]
            random_state: 内层 train/val 划分的随机种子，与外层 KFold 一致时更可复现

        Returns:
            trainloader, valloader, testloader
        """
        if split_ratio is None:
            split_ratio = [0.8, 0.1, 0.1]

        all_indices = np.arange(len(self.datalist))
        fold_indices = list(kfold.split(all_indices))

        if fold < 0 or fold >= len(fold_indices):
            raise ValueError(f"Fold index {fold} out of range (0 to {len(fold_indices)-1})")

        test_idx = fold_indices[fold][1]
        train_val_idx = fold_indices[fold][0]

        # 在非测试折上再划分 train / val：取内层 KFold 的第一折，避免重复累积 train 索引
        inner_n_splits = max(2, int(1.0 / float(split_ratio[1])))
        kfold_inner = KFold(n_splits=inner_n_splits, shuffle=True, random_state=random_state)
        tr_rel, va_rel = next(iter(kfold_inner.split(train_val_idx)))
        train_idx = [train_val_idx[i] for i in tr_rel]
        val_idx = [train_val_idx[i] for i in va_rel]

        train_data = [self.datalist[i] for i in train_idx]
        val_data = [self.datalist[i] for i in val_idx]
        test_data = [self.datalist[i] for i in test_idx]

        self.train_dataset = GraphDataset(train_data)
        self.val_dataset = GraphDataset(val_data)
        self.test_dataset = GraphDataset(test_data)

        return self.train_dataloader(batchsize), self.val_dataloader(), self.test_dataloader()

