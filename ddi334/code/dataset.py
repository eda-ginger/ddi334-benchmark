"""
DDI Dataset for DDI-Bench cluster split format (V5).

Data files (drugbank_cluster):
  train.txt        — training pairs: "h t r" (space-separated integers)
  valid_S0.txt     — transductive (warm) validation
  valid_S1.txt     — inductive (cold-single) validation
  valid_S2.txt     — inductive (cold-pair) validation
  test_S0.txt      — transductive test
  test_S1.txt      — inductive (cold-single) test
  test_S2.txt      — inductive (cold-pair) test

Format: each line is "drug1_id drug2_id label" (space-separated ints, 0-indexed).

Split terminology:
  S0 = transductive (both drugs seen in training)
  S1 = inductive (one new drug per pair)
  S2 = inductive (both drugs new)
"""
import os
from typing import Optional, Set

import numpy as np
import torch
from torch.utils.data import Dataset


# Default data path: DDI-Bench original repo
_DEFAULT_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'original_repo',
                 'DDI_Ben', 'DDI_Ben', 'data', 'drugbank_cluster')
)


class DrugBankClusterDataset(Dataset):
    """DDI-Bench DrugBank cluster-split dataset.

    Loads one split file at a time. The split name corresponds to the filename:
      'train'     -> train.txt
      'valid_S0'  -> valid_S0.txt
      'valid_S1'  -> valid_S1.txt
      'valid_S2'  -> valid_S2.txt
      'test_S0'   -> test_S0.txt
      'test_S1'   -> test_S1.txt
      'test_S2'   -> test_S2.txt

    Args:
        split: one of the split names above.
        data_dir: path to the directory containing the split files.
                  Defaults to DDI-Bench original_repo path.
        exclude_drugs: set of drug IDs to exclude (e.g. SchNet 3D-impossible drugs).
        num_classes: number of DDI relation types (86 for DrugBank Ryu-86).
    """

    def __init__(
        self,
        split: str,
        data_dir: Optional[str] = None,
        exclude_drugs: Optional[Set[int]] = None,
        num_classes: int = 86,
    ):
        self.split = split
        self.data_dir = data_dir or _DEFAULT_DATA_DIR
        self.exclude_drugs = exclude_drugs or set()
        self.num_classes = num_classes

        split_path = os.path.join(self.data_dir, f'{split}.txt')
        if not os.path.exists(split_path):
            raise FileNotFoundError(
                f"Split file not found: {split_path}\n"
                f"Expected one of: train, valid_S0, valid_S1, valid_S2, "
                f"test_S0, test_S1, test_S2")

        pairs = []
        labels = []
        with open(split_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 3:
                    continue
                d1, d2, label = int(parts[0]), int(parts[1]), int(parts[2])
                if d1 in self.exclude_drugs or d2 in self.exclude_drugs:
                    continue
                pairs.append((d1, d2))
                labels.append(label)

        self.pairs = np.array(pairs, dtype=np.int64)
        self.labels = np.array(labels, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return int(self.pairs[idx, 0]), int(self.pairs[idx, 1]), int(self.labels[idx])


class DDIBatch:
    """Simple batch container returned by collate_fn."""

    def __init__(self, drug1_ids, drug2_ids, labels,
                 mol_batch_d1=None, mol_batch_d2=None,
                 rel_subgraph_batch=None):
        self.drug1_ids = drug1_ids
        self.drug2_ids = drug2_ids
        self.labels = labels
        self.mol_batch_d1 = mol_batch_d1
        self.mol_batch_d2 = mol_batch_d2
        self.rel_subgraph_batch = rel_subgraph_batch

    def to(self, device):
        self.drug1_ids = self.drug1_ids.to(device)
        self.drug2_ids = self.drug2_ids.to(device)
        self.labels = self.labels.to(device)
        if self.mol_batch_d1 is not None:
            self.mol_batch_d1 = self.mol_batch_d1.to(device)
            self.mol_batch_d2 = self.mol_batch_d2.to(device)
        if self.rel_subgraph_batch is not None:
            self.rel_subgraph_batch = self.rel_subgraph_batch.to(device)
        return self


def build_collate_fn(
    gat_cache=None,           # list of PyG Data, indexed by drug_id (for Cell 01 GAT)
    subgraph_extractor=None,  # SubgraphExtractor instance (for Cells 09, 10)
    drug2kg_table=None,       # torch.Tensor: drug_id -> kg_entity_id
    subgraph_transform=None,  # optional callable: Data -> Data (e.g. GT._to_tiger_inputs)
):
    """Build a collate function that optionally pre-batches mol graphs and subgraphs.

    Args:
        gat_cache: if provided, pre-batch PyG Data for GAT (Cell 01).
        subgraph_extractor: if provided, pre-extract KG subgraphs (Cells 09, 10).
        drug2kg_table: required if subgraph_extractor is provided.
        subgraph_transform: optional per-Data transform applied before batching.
            Cell 09 (R-GCN subgraph): None (uses edge_index/edge_type directly).
            Cell 10 (GT KG): pass encoder._to_tiger_inputs to convert to sp_edge_*.
    """
    from torch_geometric.data import Batch as PyGBatch

    def collate_fn(samples):
        d1_list = [s[0] for s in samples]
        d2_list = [s[1] for s in samples]
        lab_list = [s[2] for s in samples]

        drug1_ids = torch.tensor(d1_list, dtype=torch.long)
        drug2_ids = torch.tensor(d2_list, dtype=torch.long)
        labels = torch.tensor(lab_list, dtype=torch.long)

        mol_batch_d1 = None
        mol_batch_d2 = None
        if gat_cache is not None:
            mol_batch_d1 = PyGBatch.from_data_list([gat_cache[d] for d in d1_list])
            mol_batch_d2 = PyGBatch.from_data_list([gat_cache[d] for d in d2_list])

        rel_subgraph_batch = None
        if subgraph_extractor is not None:
            if drug2kg_table is not None:
                kg1 = drug2kg_table[drug1_ids].tolist()
                kg2 = drug2kg_table[drug2_ids].tolist()
            else:
                kg1 = d1_list
                kg2 = d2_list
            data_list = [subgraph_extractor.extract(k1, k2)
                         for k1, k2 in zip(kg1, kg2)]
            if subgraph_transform is not None:
                data_list = [subgraph_transform(d) for d in data_list]
            rel_subgraph_batch = PyGBatch.from_data_list(data_list)

        return DDIBatch(
            drug1_ids=drug1_ids,
            drug2_ids=drug2_ids,
            labels=labels,
            mol_batch_d1=mol_batch_d1,
            mol_batch_d2=mol_batch_d2,
            rel_subgraph_batch=rel_subgraph_batch,
        )

    return collate_fn


def build_dataloaders(
    data_dir: str,
    split_types=('S0', 'S1', 'S2'),
    batch_size: int = 512,
    num_workers: int = 4,
    exclude_drugs: Optional[Set[int]] = None,
    collate_fn=None,
):
    """Build one shared train loader + per-split valid/test loaders.

    Returns:
        train_loader, valid_loaders (dict S0/S1/S2), test_loaders (dict)
    """
    from torch.utils.data import DataLoader

    if collate_fn is None:
        collate_fn = build_collate_fn()

    train_ds = DrugBankClusterDataset('train', data_dir=data_dir,
                                       exclude_drugs=exclude_drugs)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_fn,
                              pin_memory=True)

    valid_loaders = {}
    test_loaders = {}
    for s in split_types:
        valid_ds = DrugBankClusterDataset(f'valid_{s}', data_dir=data_dir,
                                           exclude_drugs=exclude_drugs)
        test_ds = DrugBankClusterDataset(f'test_{s}', data_dir=data_dir,
                                          exclude_drugs=exclude_drugs)
        valid_loaders[s] = DataLoader(valid_ds, batch_size=batch_size, shuffle=False,
                                      num_workers=num_workers, collate_fn=collate_fn,
                                      pin_memory=True)
        test_loaders[s] = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                                     num_workers=num_workers, collate_fn=collate_fn,
                                     pin_memory=True)

    return train_loader, valid_loaders, test_loaders
