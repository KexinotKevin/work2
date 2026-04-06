from torch.utils.data import Dataset, random_split
from torch_geometric.loader import DataLoader
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
        cons_thresh=0.75,
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
        return DataLoader(self.train_dataset, batch_size=batchsize, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, shuffle=False)

