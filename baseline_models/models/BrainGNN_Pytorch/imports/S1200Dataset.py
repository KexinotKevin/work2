"""
S1200 Dataset for BrainGNN
Adapted to work with HCP S1200 FC data (pFC.csv format)
"""
import os
import os.path as osp
from os import listdir
import torch
from torch_geometric.data import InMemoryDataset, Data
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class S1200Dataset(InMemoryDataset):
    """
    Dataset for HCP S1200 FC data (246 regions, BNA atlas).
    Each subject has a pFC.csv file containing a 246x246 FC matrix.
    """
    def __init__(self, root, label_col='CogFluidComp_Unadj', transform=None, pre_transform=None):
        self.root = root
        self.label_col = label_col
        super(S1200Dataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # All subjects with FC data
        raw_dir = osp.join(self.root, 'raw')
        if not osp.exists(raw_dir):
            return []
        return [f for f in listdir(raw_dir) if f.endswith('.csv')]

    @property
    def processed_file_names(self):
        return 'data_s1200.pt'

    def download(self):
        # Data already exists in raw format
        return

    def process(self):
        raw_dir = osp.join(self.root, 'raw')
        # Label file is in the original S1200 dataset directory
        labelfile = '/public/home/baitianyu/kexin/datasets/S1200/textfiles/S1200_889_CogScores.csv'

        # Load labels using numpy to avoid pandas issues
        header = ['Subject', 'Release', 'Acquisition', 'Gender', 'Age', '3T_Full_MR_Compl',
                  'T1_Count', 'T2_Count', '3T_RS-fMRI_Count', '3T_RS-fMRI_PctCompl',
                  'CogFluidComp_Unadj', 'CogFluidComp_AgeAdj', 'CogEarlyComp_Unadj',
                  'CogEarlyComp_AgeAdj', 'CogTotalComp_Unadj', 'CogTotalComp_AgeAdj',
                  'CogCrystalComp_Unadj', 'CogCrystalComp_AgeAdj']
        labels_data = np.loadtxt(labelfile, delimiter=',', dtype=str, skiprows=1)
        labels_dict = {row[0]: float(row[header.index(self.label_col)])
                       for row in labels_data}

        # Get list of all FC files
        fc_files = [f for f in listdir(raw_dir) if f.endswith('.csv')]
        fc_files.sort()

        data_list = []
        valid_subjects = []

        for fc_file in fc_files:
            subj_id = fc_file.replace('.csv', '')
            fc_path = osp.join(raw_dir, fc_file)

            # Get label for this subject
            if subj_id not in labels_dict:
                continue

            label_value = labels_dict[subj_id]
            if np.isnan(label_value):
                continue

            try:
                # Read FC matrix (246x246) using numpy to avoid pandas issues
                fc_matrix = np.loadtxt(fc_path, delimiter=',', dtype=np.float32)

                # Handle any NaN or inf values
                fc_matrix = np.nan_to_num(fc_matrix, nan=0.0, posinf=1.0, neginf=-1.0)

                # Create graph from FC matrix
                data = self._create_graph(fc_matrix, label_value, subj_id)
                data_list.append(data)
                valid_subjects.append(subj_id)
            except Exception as e:
                print(f"Error processing {subj_id}: {e}")
                continue

        if len(data_list) == 0:
            raise ValueError("No valid subjects found!")

        print(f"Processed {len(data_list)} subjects for label: {self.label_col}")

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        # Save valid subject list
        with open(osp.join(self.root, 'valid_subjects.txt'), 'w') as f:
            f.write('\n'.join(valid_subjects))

    def _create_graph(self, fc_matrix, label, subj_id):
        """
        Create a PyG Data object from FC matrix.
        For BrainGNN:
        - Node features (x): FC values (diagonal removed)
        - Edge index: upper triangle of FC matrix (excluding diagonal)
        - Edge attr: FC values as edge weights
        - pos: identity matrix for attention
        - y: label value (normalized)
        """
        num_nodes = fc_matrix.shape[0]

        # Get upper triangle indices (excluding diagonal)
        row_idx, col_idx = np.triu_indices(num_nodes, k=1)

        # Edge attributes: FC values from upper triangle
        edge_attr = fc_matrix[row_idx, col_idx].astype(np.float32)
        edge_attr = np.abs(edge_attr)  # Use absolute value for similarity

        # Edge index
        edge_index = np.stack([row_idx, col_idx])

        # Node features: use FC values as features
        # Each node gets its row from the FC matrix (excluding diagonal)
        node_features = []
        for i in range(num_nodes):
            # Get all FC values for node i except self-connection
            row = np.delete(fc_matrix[i], i)
            node_features.append(row)
        node_features = np.array(node_features, dtype=np.float32)

        # Position matrix for attention (identity-like structure)
        pos = np.eye(num_nodes, dtype=np.float32)

        # Convert to tensors
        edge_attr = torch.from_numpy(edge_attr).unsqueeze(1)
        edge_index = torch.from_numpy(edge_index).long()
        x = torch.from_numpy(node_features)
        pos = torch.from_numpy(pos)
        y = torch.tensor([label], dtype=torch.float)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=pos,
            y=y
        )

        return data

    def __repr__(self):
        return f'S1200Dataset(label={self.label_col})'


def create_raw_data(root, fc_dir, atlas='bna246'):
    """
    Create raw data directory with symlinks to FC files.
    root: where to create raw/ directory
    fc_dir: directory containing FC data (e.g., /path/to/S1200/network/bna246)
    """
    raw_dir = osp.join(root, 'raw')
    os.makedirs(raw_dir, exist_ok=True)

    subjects = [d for d in listdir(fc_dir) if osp.isdir(osp.join(fc_dir, d))]

    count = 0
    for subj in subjects:
        fc_file = osp.join(fc_dir, subj, 'FC', 'pFC.csv')
        if osp.exists(fc_file):
            # Create symlink
            link_path = osp.join(raw_dir, f'{subj}.csv')
            if not osp.exists(link_path):
                os.symlink(fc_file, link_path)
            count += 1

    print(f"Created symlinks for {count} subjects in {raw_dir}")
    return raw_dir


if __name__ == '__main__':
    # Test dataset creation
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fc_dir', type=str,
                        default='/public/home/baitianyu/kexin/datasets/S1200/network/bna246',
                        help='Directory containing FC data')
    parser.add_argument('--label_col', type=str,
                        default='CogFluidComp_Unadj',
                        help='Label column to use')
    args = parser.parse_args()

    root = '/public/home/baitianyu/kexin/projects/work2/baseline_models/models/BrainGNN_Pytorch/data_s1200'

    # Create raw data symlinks
    create_raw_data(root, args.fc_dir)

    # Create dataset
    dataset = S1200Dataset(root=root, label_col=args.label_col)

    print(f"Dataset size: {len(dataset)}")
    print(f"Data sample: {dataset[0]}")
