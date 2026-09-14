"""All DDI review-encoder modules in one place (단일 파일 통합).

frozen-vector(05/06/12/13/14) / molecule structural(01-04,07) / relation-KG(08-11)
인코더를 한 파일에서 관리. 공유 base FrozenVectorEncoder도 여기 있어 파일간 import/중복 없음.
동작/구조/입력은 v5 기준 그대로 (병합은 배치만 변경).
"""

from collections import defaultdict
from rdkit import Chem
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, SAGPooling, global_add_pool, global_mean_pool
from torch_geometric.nn import RGCNConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax as pyg_softmax
from typing import Dict, List, Optional, Tuple
from typing import Dict, Optional
import math
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# FROZEN-VECTOR ENCODERS  (precompute vector -> MLP; cells 05/06/12/13/14)
# ============================================================================

class FrozenVectorEncoder(nn.Module):
    """Generic frozen precomputed-vector encoder: table lookup -> linear projection.

    Input: {drug_id (int): Tensor[bert_dim]} — precomputed externally (frozen).
    Pipeline: emb[drug_id] -> Linear(bert_dim->hidden) -> ReLU -> Dropout.
    Modality-specific subclasses only set docstring + default dim (no body change).
    bert_dim = 입력 feature 차원.
    """

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.2,
                 bert_dim: int = 768, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.bert_dim = bert_dim
        self.project = nn.Sequential(
            nn.Linear(bert_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self._embedding_table: Optional[torch.Tensor] = None

    def set_features(self, embeddings: Dict[int, torch.Tensor]):
        """Build lookup table from drug_id -> precomputed embedding."""
        max_id = max(embeddings.keys())
        table = torch.zeros(max_id + 1, self.bert_dim)
        for did, emb in embeddings.items():
            table[did] = emb
        self._embedding_table = table

    def forward(self, drug_ids: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        if self._embedding_table.device != device:
            self._embedding_table = self._embedding_table.to(device)
        emb = self._embedding_table[drug_ids.long()]
        return self.project(emb)


# ══════════════════════════════════════════════════════════════
# 셀별 named 서브클래스 — 본문 동일(위 base), docstring + 기본 dim만 다름.
# 입력 데이터(precompute)만 다르고 연산(MLP)은 같은 것들을 한 곳에 통칭.
# ══════════════════════════════════════════════════════════════

class FPJaccardEncoder(FrozenVectorEncoder):
    """Cell 05 (MOL): Morgan FP pairwise Jaccard -> PCA -> MLP refinement.

    Source: DDIMDL (Deng et al., 2020) — Morgan FP Jaccard similarity + PCA.
    Fair-exp adaptation: Jaccard+PCA를 train(Dk) 약물에만 fit(누수 제거)하고 외부
      precompute(`precompute_pca.py`); dim = #Dk. set_features는 [N, dim] (bert_dim=PCA dim).
    """

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.2,
                 bert_dim: int = 270, **kwargs):
        super().__init__(hidden_dim=hidden_dim, dropout=dropout, bert_dim=bert_dim, **kwargs)


class ChemBERTaEncoder(FrozenVectorEncoder):
    """Cell 06 (MOL): ChemBERTa frozen [CLS] -> linear projection.

    Source: LLME (Im & Ko, 2025) / HKG-DDIE (Asada et al., 2023) —
            DeepChem/ChemBERTa-77M-MLM (PubChem 77M SMILES), frozen [CLS] 384-d.
    Fair-exp adaptation: [CLS]를 frozen으로 고정하고 단일 linear re-weighting만 학습
      (multi-layer MLP 없음, LLME 따름); 임베딩은 외부 precompute.
    """

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.2,
                 bert_dim: int = 384, **kwargs):
        super().__init__(hidden_dim=hidden_dim, dropout=dropout, bert_dim=bert_dim, **kwargs)


class InteractionJaccardEncoder(FrozenVectorEncoder):
    """Cell 12 (REL): HetioNet bio interaction profile Jaccard -> PCA -> MLP.

    Source: DDIMDL (Deng et al., 2020) — multi-attribute interaction-profile Jaccard + PCA.
    Fair-exp adaptation: HetioNet bio 이웃(drug-gene/pathway 등) 프로파일로 pairwise
      Jaccard -> PCA(`precompute_pca.py`)를 train(Dk)에만 fit(누수 제거);
      set_features는 [N, dim] (bert_dim = PCA dim).
    """

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.2,
                 bert_dim: int = 270, **kwargs):
        super().__init__(hidden_dim=hidden_dim, dropout=dropout, bert_dim=bert_dim, **kwargs)


class BioBERTEncoder(FrozenVectorEncoder):
    """Cell 13/14 (DOC): BioBERT frozen [CLS] -> linear projection.

    Source: LLME (Im & Ko, 2025) — dmis-lab/biobert-base-cased-v1.1 frozen [CLS] 768-d.
    Fair-exp adaptation: [CLS]를 frozen으로 고정하고 단일 linear projection만 학습
      (multi-layer MLP 없음); 텍스트 임베딩은 외부 precompute.
      Cell 13 = drug description / Cell 14 = drug name 입력.
    """

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.2,
                 bert_dim: int = 768, **kwargs):
        super().__init__(hidden_dim=hidden_dim, dropout=dropout, bert_dim=bert_dim, **kwargs)


# 하위호환 alias (구 v5 train.py가 BioBERTFrozenEncoder를 generic frozen으로 import)
BioBERTFrozenEncoder = BioBERTEncoder

# ============================================================================
# MOLECULE STRUCTURAL ENCODERS + helpers  (cells 01 GAT / 02 SchNet / 03 GT-mol / 04 FP / 07 CNN)
# ============================================================================

# ── Shared: SMILES character tokenizer ──
# DeepPurpose standard 63-char vocab + stereochemistry chars (/@\)
# Ref: DeepPurpose (Huang et al.), SSF-DDI (vocab_size=65)
SMILES_CHARS = [
    '?',  # index 0: unknown/padding
    '#', '%', ')', '(', '+', '-', '.', '1', '0', '3', '2', '5', '4',
    '7', '6', '9', '8', '=', 'A', 'C', 'B', 'E', 'D', 'G', 'F', 'I',
    'H', 'K', 'M', 'L', 'O', 'N', 'P', 'S', 'R', 'U', 'T', 'W', 'V',
    'Y', '[', 'Z', ']', '_', 'a', 'c', 'b', 'e', 'd', 'g', 'f', 'i',
    'h', 'm', 'l', 'o', 'n', 's', 'r', 'u', 't', 'y',
    '/', '@', '\\',  # stereochemistry (not in DeepPurpose, present in DDI data)
]
CHAR2IDX = {c: i for i, c in enumerate(SMILES_CHARS)}
VOCAB_SIZE = len(SMILES_CHARS)
MAX_SMILES_LEN = 100


def tokenize_smiles(smiles: str, max_len: int = MAX_SMILES_LEN) -> list:
    tokens = [CHAR2IDX.get(c, 0) for c in smiles[:max_len]]
    tokens += [0] * (max_len - len(tokens))
    return tokens


# ── Atom feature constants ──
ATOM_SYMBOLS = [
    'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
    'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag',
    'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni',
    'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'X'
]  # 44 symbols (X = unknown). SSI-DDI utils.py + TIGER utils.py consistent.
ATOM_SYMBOL_MAP = {s: i for i, s in enumerate(ATOM_SYMBOLS)}
HYBRIDIZATION_LIST = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
NUM_HS_RANGE = list(range(11))
VALENCE_RANGE = list(range(11))
MAX_DEGREE = 10

# Cell 01 (GAT): SSI-DDI original 55-dim
SSI_DDI_FEAT_DIM = 55   # 44 + 1 + 1 + 1 + 5 + 1 + 1 + 1(padding) = 55
# Cell 03 (GT):  TIGER original 67-dim
TIGER_FEAT_DIM = 67     # 44 + 11 + 11 + 1 = 67
# Legacy alias
ATOM_FEAT_DIM = SSI_DDI_FEAT_DIM

# Number of bond types for TIGER-style GT
NUM_BOND_TYPES = 22


def _onehot(value, allowable_set):
    """One-hot list, last bin used as fallback."""
    if value not in allowable_set:
        value = allowable_set[-1]
    return [int(v == value) for v in allowable_set]


def atom_features_ssi_ddi(atom) -> list:
    """55-dim atom features following SSI-DDI original (Nyamabo 2021).
    44(symbol) + 1(degree) + 1(Hs) + 1(valence) + 5(hybridization) + 1(aromatic) + 1(fcharge) = 54 -> pad to 55.
    """
    symbol = atom.GetSymbol()
    sym_idx = ATOM_SYMBOL_MAP.get(symbol, ATOM_SYMBOL_MAP['X'])
    symbol_onehot = [0] * 44
    symbol_onehot[sym_idx] = 1
    feats = symbol_onehot + [
        atom.GetDegree(),
        atom.GetTotalNumHs(),
        atom.GetImplicitValence(),
    ]
    feats += [int(atom.GetHybridization() == h) for h in HYBRIDIZATION_LIST]
    feats += [int(atom.GetIsAromatic()), atom.GetFormalCharge()]
    feats += [0] * (SSI_DDI_FEAT_DIM - len(feats))  # pad to 55
    return feats


def atom_features_tiger(atom) -> list:
    """67-dim atom features following TIGER original (utils.py:203-212).
    44(symbol) + 11(TotalNumHs one-hot) + 11(ImplicitValence one-hot) + 1(IsAromatic) = 67.
    """
    symbol = atom.GetSymbol()
    sym_idx = ATOM_SYMBOL_MAP.get(symbol, ATOM_SYMBOL_MAP['X'])
    symbol_onehot = [0] * 44
    symbol_onehot[sym_idx] = 1
    feats = symbol_onehot
    feats += _onehot(atom.GetTotalNumHs(), NUM_HS_RANGE)
    feats += _onehot(atom.GetImplicitValence(), VALENCE_RANGE)
    feats += [int(atom.GetIsAromatic())]
    return feats  # 67 total


# Legacy aliases
atom_features_55d = atom_features_ssi_ddi


def mol_to_graph_55d(smiles: str) -> dict:
    """Convert SMILES to molecular graph with SSI-DDI 55-dim atom features (Cell 01)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            'x': np.zeros((1, SSI_DDI_FEAT_DIM), dtype=np.float32),
            'edge_index': np.zeros((2, 0), dtype=np.int64),
        }
    atoms = []
    for atom in mol.GetAtoms():
        feat = atom_features_ssi_ddi(atom)
        atoms.append(feat)
    x = np.array(atoms, dtype=np.float32)
    edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.append([i, j])
        edges.append([j, i])
    edge_index = np.array(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
    return {'x': x, 'edge_index': edge_index}


def mol_to_graph_tiger(smiles: str, max_distance: int = 8) -> dict:
    """Convert SMILES to TIGER-style mol graph (Cell 03).

    Returns a dict with:
      - `x`: [N, 67] atom features
      - `edge_index`: [2, 2*B] direct-bond edges (both directions)
      - `sp_edge_index`: [2, E] all atom pairs (i, j) with graph distance <= max_distance
      - `sp_value`: [E] float graph distance
      - `sp_edge_rel`: [E] int rel type:
            0..21 = direct bond type (RDKit bond.GetBondType() enum int)
            22+d  = indirect pair, where d is graph distance >= 2
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            'x': np.zeros((1, TIGER_FEAT_DIM), dtype=np.float32),
            'edge_index': np.zeros((2, 0), dtype=np.int64),
            'sp_edge_index': np.zeros((2, 0), dtype=np.int64),
            'sp_value': np.zeros((0,), dtype=np.float32),
            'sp_edge_rel': np.zeros((0,), dtype=np.int64),
        }
    n = mol.GetNumAtoms()
    atoms = []
    for atom in mol.GetAtoms():
        feat = atom_features_tiger(atom)
        atoms.append(feat)
    x = np.array(atoms, dtype=np.float32)

    # Direct-bond edge_index (both directions)
    direct_edges = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        direct_edges.append([i, j])
        direct_edges.append([j, i])
    edge_index = (np.array(direct_edges, dtype=np.int64).T
                  if direct_edges else np.zeros((2, 0), dtype=np.int64))

    # Direct bond type per pair (i, j). -1 = no direct bond.
    bond_type = -np.ones((n, n), dtype=np.int64)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = int(bond.GetBondType())
        bt = min(bt, NUM_BOND_TYPES - 1)
        bond_type[i, j] = bt
        bond_type[j, i] = bt

    # Graph topology distance matrix
    dist = Chem.GetDistanceMatrix(mol).astype(np.float32)

    edges, rel_types, sp_vals = [], [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = float(dist[i, j])
            if not np.isfinite(d) or d > max_distance:
                continue
            d_int = int(d)
            if bond_type[i, j] >= 0:
                rt = int(bond_type[i, j])
            else:
                rt = NUM_BOND_TYPES + d_int
            edges.append([i, j])
            rel_types.append(rt)
            sp_vals.append(d)
    sp_edge_index = (np.array(edges, dtype=np.int64).T
                     if edges else np.zeros((2, 0), dtype=np.int64))
    sp_edge_rel = np.array(rel_types, dtype=np.int64)
    sp_value = np.array(sp_vals, dtype=np.float32)
    return {
        'x': x,
        'edge_index': edge_index,
        'sp_edge_index': sp_edge_index,
        'sp_value': sp_value,
        'sp_edge_rel': sp_edge_rel,
    }


# ══════════════════════════════════════════════════════════════
# Cell 04: Morgan FP — Ref: MRLF-DDI (radius=2, 1024-bit)
# ══════════════════════════════════════════════════════════════

class MorganFPEncoder(nn.Module):
    """Cell 04 (MOL): Morgan Fingerprint (ECFP4) -> 2-layer MLP.

    Source: MRLF-DDI (Zhong et al., 2025) — Morgan 1024-bit, radius=2, MLP -> 128d.
    Fair-exp adaptation: 원 모델의 multi-representation 중 FP branch만 사용
      (다른 모달리티/분기 제거) 하여 fingerprint 단일 인코더로 통제 비교.
    """

    def __init__(self, hidden_dim: int = 128, fp_dim: int = 1024, dropout: float = 0.2, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fp_dim = fp_dim
        self.encoder = nn.Sequential(
            nn.Linear(fp_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self._fp_table: Optional[torch.Tensor] = None  # [max_id+1, fp_dim]

    def set_features(self, fingerprints: Dict[int, np.ndarray]):
        """Pre-build tensor table for O(1) batch lookup."""
        max_id = max(fingerprints.keys())
        table = np.zeros((max_id + 1, self.fp_dim), dtype=np.float32)
        for did, fp in fingerprints.items():
            table[did] = fp
        self._fp_table = torch.from_numpy(table)

    def forward(self, drug_ids: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        if self._fp_table.device != device:
            self._fp_table = self._fp_table.to(device)
        x = self._fp_table[drug_ids.long()]  # [B, fp_dim]
        return self.encoder(x)


class BioProfileMLPEncoder(nn.Module):
    """Cell 15 (REL): raw HetioNet bio-neighbor multi-hot vector -> 2-layer MLP.

    Source: MDF-SA-DDI (Lin et al., 2022) — target+enzyme raw binary vector fed
      directly into a downstream network (CNN+Autoencoder in the original paper).
    Fair-exp adaptation: the raw vector -> plain MLP step is kept, but the
      downstream network is replaced with the same 2-layer MLP used by Cell 04
      (MorganFPEncoder), so that the two raw-vector-to-MLP cells (Molecule
      fingerprint / Relation bio profile) share one architecture and differ only
      in their input vector (user decision, 2026-09-08). No Jaccard similarity or
      PCA step, unlike Cell 12 (InteractionJaccardEncoder), which reduces the same
      raw profile through similarity+PCA first.
    Input (build_bio_raw.py): 554-dim binary vector = which of 554 non-drug HetioNet
      entities (486 target Gene / 68 Disease) each drug connects to. Restricted to
      Gene and Disease only (Pharmacologic Class and Compound dropped, user decision
      2026-09-08): Compound neighbors turned out to be connected via CrC (Compound-
      resembles-Compound), a structural-similarity relation that CLAUDE.md's modality
      rule assigns to Molecule, not Relation ("KG containing only molecular
      similarity edges = Molecule") -- confirmed against the original HetioNet SIF
      edge file; Pharmacologic Class was dropped to keep the profile aligned with the
      target-only focus of the cited literature (DDIMDL/MDF-SA-DDI). Two further
      exclusions match build_bio_profile.py's LLM-facing text profile (not applied by
      precompute_pca.py's Cell 12): (1) Compound-causes-Side Effect edges (relation
      21) are dropped, since this HetioNet relation shares its vocabulary with the
      DDI-334 prediction label (adverse events) and would leak it; (2) Gene neighbors
      are restricted to direct target binding (Compound-binds-Gene, relation 6) --
      indirect up/downregulation relations are dropped.
    """

    def __init__(self, hidden_dim: int = 128, fp_dim: int = 554, dropout: float = 0.2, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fp_dim = fp_dim
        self.encoder = nn.Sequential(
            nn.Linear(fp_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self._fp_table: Optional[torch.Tensor] = None  # [max_id+1, fp_dim]

    def set_features(self, profiles: Dict[int, torch.Tensor]):
        """Pre-build tensor table for O(1) batch lookup."""
        max_id = max(profiles.keys())
        table = torch.zeros(max_id + 1, self.fp_dim, dtype=torch.float32)
        for did, v in profiles.items():
            table[did] = v
        self._fp_table = table

    def forward(self, drug_ids: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        if self._fp_table.device != device:
            self._fp_table = self._fp_table.to(device)
        x = self._fp_table[drug_ids.long()]  # [B, fp_dim]
        return self.encoder(x)


# ══════════════════════════════════════════════════════════════
# Cell 01: GAT — Ref: SSI-DDI (Nyamabo 2021)
#    Multi-block GAT with SAGPooling as soft-attention readout.
# ══════════════════════════════════════════════════════════════

class GATEncoder(nn.Module):
    """Cell 01 (MOL): Multi-block GAT + SAGPool on molecular graph.

    Source: SSI-DDI (Nyamabo et al., 2021) — 4 blocks, 2 heads, 32d/head, SAGPool readout.
    Fair-exp adaptation: SSI-DDI의 substructure relation embedding(DDI type-specific
      co-attention) 제거하고 약물 임베딩만 추출; 노드 피처를 Cell 03(GT)과 동일한
      TIGER 67-d로 통일(`atom_feat_dim=67`, ddi334 호출부) → 두 그래프 셀의 차이를
      attention 방식으로만 통제; block 간 graph pruning 안 함, per-block readout 평균.
    """

    def __init__(self, hidden_dim: int = 128, num_blocks: int = 4,
                 heads: int = 2, atom_feat_dim: int = SSI_DDI_FEAT_DIM,
                 max_degree: int = MAX_DEGREE,
                 dropout: float = 0.2, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.max_degree = max_degree
        self.atom_feat_dim = atom_feat_dim

        # Initial projection + norm
        self.initial_norm = nn.LayerNorm(atom_feat_dim)
        self.atom_embed = nn.Linear(atom_feat_dim, hidden_dim)
        self.degree_encoder = nn.Embedding(max_degree + 1, hidden_dim, padding_idx=0)

        # Per-block: GATConv + SAGPool readout + LayerNorm
        head_out_feats = hidden_dim // heads
        self.convs = nn.ModuleList()
        self.readouts = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.projs = nn.ModuleList()

        in_dim = hidden_dim
        for i in range(num_blocks):
            self.convs.append(GATConv(in_dim, head_out_feats,
                                      heads=heads, dropout=dropout, concat=True))
            out_dim = head_out_feats * heads
            self.readouts.append(SAGPooling(out_dim, min_score=-1))
            self.norms.append(nn.LayerNorm(out_dim))
            self.projs.append(nn.Linear(out_dim, hidden_dim))
            in_dim = out_dim

        self.dropout = dropout
        self._data_cache: Optional[list] = None

    def set_features(self, mol_graphs: Dict[int, dict]):
        """Pre-build PyG Data objects for O(1) lookup per drug."""
        max_id = max(mol_graphs.keys())
        dummy = Data(
            x=torch.zeros((1, self.atom_feat_dim), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
        )
        self._data_cache = [dummy] * (max_id + 1)
        for did, g in mol_graphs.items():
            self._data_cache[did] = Data(
                x=torch.tensor(g['x'], dtype=torch.float32),
                edge_index=torch.tensor(g['edge_index'], dtype=torch.long),
            )

    def _build_batch(self, drug_ids: torch.Tensor) -> Batch:
        device = next(self.parameters()).device
        ids = drug_ids.cpu().tolist()
        data_list = [self._data_cache[did] for did in ids]
        return Batch.from_data_list(data_list).to(device)

    def forward(self, drug_ids: torch.Tensor = None,
                mol_batch: Batch = None) -> torch.Tensor:
        from torch_geometric.utils import degree as pyg_degree
        if mol_batch is not None:
            batch = mol_batch.to(next(self.parameters()).device)
        else:
            batch = self._build_batch(drug_ids)

        x = self.initial_norm(batch.x)
        x = self.atom_embed(x)
        edge_index = batch.edge_index
        batch_vec = batch.batch
        x_degree = pyg_degree(edge_index[1], batch.x.size(0),
                              dtype=torch.long).clamp(max=self.max_degree)
        x = x + self.degree_encoder(x_degree)

        block_embeds = []
        for conv, readout, norm, proj in zip(
                self.convs, self.readouts, self.norms, self.projs):
            x = conv(x, edge_index)
            att_x, _, _, att_batch, _, _ = readout(
                x, edge_index, batch=batch_vec)
            graph_emb = global_add_pool(att_x, att_batch)  # [B, out_dim]
            block_embeds.append(proj(graph_emb))
            x = F.elu(norm(x))
            x = F.dropout(x, p=self.dropout, training=self.training)

        out = torch.stack(block_embeds, dim=0).mean(dim=0)  # [B, hidden_dim]
        return out


# NOTE: frozen-vector 셀(05 FP-Jaccard / 06 ChemBERTa / 12 Interaction-Jaccard /
#       13·14 BioBERT)은 연산이 동일(precompute 벡터 -> MLP)하므로
#       models/frozen_encoder.py에 통합. mol_encoders는 구조적 인코더만 보유.


# ══════════════════════════════════════════════════════════════
# Cell 07: SMILES CNN — Ref: SSF-DDI (2024)
#    Progressive kernels [4,6,8], filters [40,80,160], char-level
# ══════════════════════════════════════════════════════════════

class SMILESCNNEncoder(nn.Module):
    """Cell 07 (MOL): 1D CNN on SMILES character sequence.

    Source: SSF-DDI (2024, BMC Bioinformatics) — char embed 64d, Conv [4,6,8]x[40,80,160],
            MaxPool1d(85), char-level tokenizer (vocab=65), max_len=100.
    Fair-exp adaptation: SSF-DDI의 substructure cross-attention(SSI) 분기 제거하고
      SMILES CNN branch만 사용; SMILES는 길이 100 고정(짧으면 0-pad, 길면 절단) — 가변 X.
    """

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.2,
                 embed_dim: int = 64, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(embed_dim, 40,  kernel_size=4)
        self.conv2 = nn.Conv1d(40,        80,  kernel_size=6)
        self.conv3 = nn.Conv1d(80,        160, kernel_size=8)
        # MaxPool1d(85): 100 - 4 - 6 - 8 + 3 = 85
        self.pool = nn.MaxPool1d(kernel_size=85)
        self.project = nn.Sequential(
            nn.Linear(160, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self._token_table: Optional[torch.Tensor] = None  # [max_id+1, MAX_SMILES_LEN]

    def set_features(self, smiles_dict: Dict[int, str]):
        """Pre-tokenize all SMILES into tensor table for O(1) batch lookup."""
        max_id = max(smiles_dict.keys())
        table = np.zeros((max_id + 1, MAX_SMILES_LEN), dtype=np.int64)
        for did, smi in smiles_dict.items():
            table[did] = tokenize_smiles(smi)
        self._token_table = torch.from_numpy(table)

    def forward(self, drug_ids: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        if self._token_table.device != device:
            self._token_table = self._token_table.to(device)
        x = self._token_table[drug_ids.long()]  # [B, L=100]
        x = self.embed(x).transpose(1, 2)       # [B, 64, 100]
        x = F.relu(self.conv1(x))               # [B, 40, 97]
        x = F.relu(self.conv2(x))               # [B, 80, 92]
        x = F.relu(self.conv3(x))               # [B, 160, 85]
        x = self.pool(x).squeeze(2)             # [B, 160]
        return self.project(x)


# ══════════════════════════════════════════════════════════════
# Cell 02: 3D SchNet — Ref: 3DGT-DDI / MDJCL-DDI
#    hidden=128, filters=128, num_interactions=6, gaussians=50,
#    cutoff=10.0, readout='add'
# ══════════════════════════════════════════════════════════════


class _SchNetEmbedder(object):
    """Lazy alias holder so the import is deferred until first instantiation."""
    _cls = None

    @classmethod
    def get(cls):
        if cls._cls is None:
            from torch_geometric.nn.models import SchNet as _SchNet
            from torch_geometric.utils import scatter as _scatter

            class SchNetEmbed(_SchNet):
                """PyG SchNet variant returning per-graph embeddings of
                dim `hidden_channels // 2` (post-lin1+act, pre-lin2)."""

                def forward(self, z, pos, batch=None):
                    batch = torch.zeros_like(z) if batch is None else batch
                    h = self.embedding(z)
                    edge_index, edge_weight = self.interaction_graph(pos, batch)
                    edge_attr = self.distance_expansion(edge_weight)
                    for interaction in self.interactions:
                        h = h + interaction(h, edge_index, edge_weight, edge_attr)
                    h = self.lin1(h)
                    h = self.act(h)
                    reduce = self.readout if isinstance(self.readout, str) else 'add'
                    return _scatter(h, batch, dim=0, reduce=reduce)
            cls._cls = SchNetEmbed
        return cls._cls


class SchNetEncoder(nn.Module):
    """Cell 02 (MOL): 3D SchNet on RDKit MMFF94-optimized conformers.

    Source: 3DGT-DDI (He et al., 2022) / MDJCL-DDI — 3D atom coords + SchNet.
    Fair-exp adaptation: 3D 좌표를 RDKit ETKDG+MMFF94로 사전계산(`schnet_coords.py`,
      모델 외부에서 1회) 후 frozen 입력; drug 1050(Fe-CN complex)은 MANUAL_OCTAHEDRAL
      좌표; 좌표 실패/결측 약물은 단일-H dummy로 대체; readout='add'. (DDI-334: 334 전부 커버)
    """

    def __init__(self, hidden_dim: int = 128, num_filters: int = 128,
                 num_interactions: int = 6, num_gaussians: int = 50,
                 cutoff: float = 10.0, dropout: float = 0.2, **kwargs):
        super().__init__()
        SchNetEmbed = _SchNetEmbedder.get()
        self.hidden_dim = hidden_dim
        self.cutoff = cutoff
        self.schnet = SchNetEmbed(
            hidden_channels=hidden_dim,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            readout='add',
        )
        self.project = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self._z_cache: Optional[list] = None
        self._pos_cache: Optional[list] = None
        self._drop_set: set = set()

    def set_features(self, coords_dict: Dict[int, dict],
                     drop_drug_ids: Optional[list] = None):
        """coords_dict: {drug_id: {'z': np.ndarray (N,), 'pos': np.ndarray (N, 3)}}.
        drop_drug_ids: drug_ids to replace with single-atom H dummy (normally empty).
        """
        if drop_drug_ids is not None:
            self._drop_set = set(drop_drug_ids)
        max_id = max(coords_dict.keys()) if coords_dict else -1
        if drop_drug_ids:
            max_id = max(max_id, max(drop_drug_ids))
        self._z_cache = [None] * (max_id + 1)
        self._pos_cache = [None] * (max_id + 1)
        for did, c in coords_dict.items():
            self._z_cache[did] = torch.tensor(c['z'], dtype=torch.long)
            self._pos_cache[did] = torch.tensor(c['pos'], dtype=torch.float32)

    def forward(self, drug_ids: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        ids = drug_ids.cpu().tolist()
        z_list, pos_list, batch_list = [], [], []
        for b_idx, did in enumerate(ids):
            cached_z = self._z_cache[did] if (self._z_cache and did < len(self._z_cache)) else None
            if did in self._drop_set or cached_z is None:
                # Single-atom hydrogen dummy for failed/missing drugs.
                z_list.append(torch.tensor([1], dtype=torch.long))
                pos_list.append(torch.zeros(1, 3, dtype=torch.float32))
                batch_list.append(torch.full((1,), b_idx, dtype=torch.long))
            else:
                z = cached_z
                pos = self._pos_cache[did]
                z_list.append(z)
                pos_list.append(pos)
                batch_list.append(torch.full((z.size(0),), b_idx, dtype=torch.long))
        z = torch.cat(z_list).to(device)
        pos = torch.cat(pos_list).to(device)
        batch = torch.cat(batch_list).to(device)
        h = self.schnet(z, pos, batch)              # [B, hidden_dim // 2]
        return self.project(h)


# ══════════════════════════════════════════════════════════════
# Cell 03: Graph Transformer (TIGER-style)
#    Ref: TIGER (AAAI 2024) + RISE-DDI (AAAI 2026)
#    Generic _TigerStyleGT is also reused by Cell 10 (KG GT).
# ══════════════════════════════════════════════════════════════


class _TigerSpatialEncoding(nn.Module):
    """TIGER SpatialEncoding: scalar distance -> scalar attention bias.

    Source: TIGER model/GraphTransformer.py:60-79
    """

    def __init__(self, dim_model: int):
        super().__init__()
        self.fnn = nn.Sequential(
            nn.Linear(1, dim_model), nn.ReLU(),
            nn.Linear(dim_model, 1), nn.ReLU(),
        )

    def forward(self, sp_value: torch.Tensor) -> torch.Tensor:
        return self.fnn(sp_value.unsqueeze(-1))   # [E, 1]


class _TigerMultiheadAttention(MessagePassing):
    """TIGER MultiheadAttention(MessagePassing) faithful re-implementation.

    Source: TIGER model/GraphTransformer.py:88-189 + RISE-DDI fork (identical).
    - 4-th root scaling: 1 / sqrt(sqrt(depth)) (TIGER line 171)
    - Linear attention normalization: weight = (Q+rel)*(K+rel) / (Q * sum_K)
    - Numerical safety: denominator + 1e-6 epsilon
    """

    def __init__(self, dim_model: int, num_heads: int,
                 rel_encoder: nn.Embedding,
                 spatial_encoder: _TigerSpatialEncoding):
        super().__init__(aggr='add')
        assert dim_model % num_heads == 0
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.depth = dim_model // num_heads
        self.rel_embedding = rel_encoder
        self.spatial_encoding = spatial_encoder
        self.wq = nn.Linear(dim_model, dim_model)
        self.wk = nn.Linear(dim_model, dim_model)
        self.wv = nn.Linear(dim_model, dim_model)
        self.dense = nn.Linear(dim_model, dim_model)

    @staticmethod
    def _denominator(qs, ks):
        """TIGER denominator() (GraphTransformer.py:149-153) — Q * sum_K per head."""
        ks_sum = ks.sum(dim=0)                             # [H, P]
        return torch.einsum("nhp,hp->nh", qs, ks_sum)      # [N, H]

    def forward(self, x, sp_edge_index, sp_value, sp_edge_rel):
        rel_emb = self.rel_embedding(sp_edge_rel)         # [E, D]
        q = self.wq(x); k = self.wk(x); v = self.wv(x)    # [N, D]

        H, P = self.num_heads, self.depth
        q_h = q.view(-1, H, P)
        k_h = k.view(-1, H, P)
        v_h = v.view(-1, H, P)

        row, col = sp_edge_index
        q_end = (q[col] + rel_emb).view(-1, H, P)         # [E, H, P]
        k_start = (k[row] + rel_emb).view(-1, H, P)       # [E, H, P]

        scale = 1.0 / math.sqrt(math.sqrt(P))
        edge_attn = torch.einsum("ehp,ehp->eh", q_end, k_start) * scale  # [E, H]
        edge_attn = edge_attn + self.spatial_encoding(sp_value)           # [E, H]

        denom = self._denominator(q_h, k_h)                               # [N, H]
        attn_w = edge_attn / (denom[col] + 1e-6)                          # [E, H]

        v_flat = v_h.reshape(-1, H * P)                                   # [N, H*P]
        out = self.propagate(edge_index=sp_edge_index, x=v_flat,
                             edge_weight=attn_w)                          # [N, H*P]
        return self.dense(out)

    def message(self, x_j, edge_weight):
        H, P = self.num_heads, self.depth
        x_split = x_j.view(-1, H, P)
        msg = x_split * edge_weight.unsqueeze(-1)
        return msg.reshape(-1, H * P)


class _TigerGTLayer(nn.Module):
    """One TIGER GraphTransformerEncode layer (pre-LN MHA + pre-LN FFN).

    Source: TIGER model/GraphTransformer.py:18-55.
    """

    def __init__(self, dim_model: int, num_heads: int,
                 rel_encoder: nn.Embedding,
                 spatial_encoder: _TigerSpatialEncoding,
                 dropout: float = 0.2):
        super().__init__()
        self.attn = _TigerMultiheadAttention(
            dim_model, num_heads, rel_encoder, spatial_encoder)
        self.ffn = nn.Sequential(
            nn.Linear(dim_model, dim_model * 2),
            nn.ReLU(),
            nn.Linear(dim_model * 2, dim_model),
        )
        self.norm1 = nn.LayerNorm(dim_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim_model, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, sp_edge_index, sp_value, sp_edge_rel):
        attn_out = self.attn(self.norm1(x), sp_edge_index, sp_value, sp_edge_rel)
        x = x + self.dropout1(attn_out)
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout2(ffn_out)
        return x


class _TigerStyleGT(nn.Module):
    """Stack of TIGER GT layers with shared rel_encoder + spatial_encoder.

    Source: TIGER model/GraphTransformer.py:191-247.
    Generic — used by both Cell 03 (mol, num_rel~31) and Cell 10 (KG, num_rel~32).
    """

    def __init__(self, dim_model: int, num_layers: int,
                 num_heads: int, num_rel: int, dropout: float = 0.2):
        super().__init__()
        self.rel_encoder = nn.Embedding(num_rel, dim_model)
        self.spatial_encoder = _TigerSpatialEncoding(dim_model)
        self.layers = nn.ModuleList([
            _TigerGTLayer(dim_model, num_heads,
                          self.rel_encoder, self.spatial_encoder, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, sp_edge_index, sp_value, sp_edge_rel):
        for layer in self.layers:
            x = layer(x, sp_edge_index, sp_value, sp_edge_rel)
        return x


class GraphTransformerMolEncoder(nn.Module):
    """Cell 03 (MOL): TIGER-style molecular Graph Transformer (atom+degree+bond+spatial).

    Source: TIGER (Su et al., 2024) — mol-side Graph Transformer.
    Fair-exp adaptation: TIGER에서 분자측 GT만 추출(KG/substructure 분기 제거)하여
      mol 단일 인코더로 통제; TIGER GT attention은 repo의 linear/Performer 공식 사용
      (paper eq(2) softmax 아님; _TigerMultiheadAttention 참조); 67-d TIGER atom feature,
      degree는 nn.Embedding 별도 처리.
    Input: set_features({drug_id: mol_to_graph_tiger(smiles)})로 cache 구축.
    """

    def __init__(self, hidden_dim: int = 128, num_layers: int = 2,
                 num_heads: int = 4, max_distance: int = 8,
                 atom_feat_dim: int = TIGER_FEAT_DIM,
                 max_degree: int = MAX_DEGREE,
                 dropout: float = 0.2, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_distance = max_distance
        self.max_degree = max_degree
        # 0..NUM_BOND_TYPES-1 = bonds; NUM_BOND_TYPES..NUM_BOND_TYPES+max_distance = distances
        num_rel = NUM_BOND_TYPES + max_distance + 1

        self.atom_norm = nn.LayerNorm(atom_feat_dim)
        self.atom_embed = nn.Linear(atom_feat_dim, hidden_dim)
        self.degree_encoder = nn.Embedding(max_degree + 1, hidden_dim, padding_idx=0)
        self.gt = _TigerStyleGT(hidden_dim, num_layers, num_heads, num_rel,
                                dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self._cache: Optional[list] = None

    def set_features(self, mol_graphs: Dict[int, dict]):
        max_id = max(mol_graphs.keys())
        dummy = Data(
            x=torch.zeros((1, ATOM_FEAT_DIM), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            sp_edge_index=torch.zeros((2, 0), dtype=torch.long),
            sp_value=torch.zeros((0,), dtype=torch.float32),
            sp_edge_rel=torch.zeros((0,), dtype=torch.long),
        )
        self._cache = [dummy] * (max_id + 1)
        for did, g in mol_graphs.items():
            if 'edge_index' in g:
                edge_index = torch.tensor(g['edge_index'], dtype=torch.long)
            else:
                spi = np.asarray(g['sp_edge_index'])
                spv = np.asarray(g['sp_value'])
                mask = spv == 1.0
                edge_index = torch.tensor(spi[:, mask] if spi.ndim == 2 else spi,
                                          dtype=torch.long)
            self._cache[did] = Data(
                x=torch.tensor(g['x'], dtype=torch.float32),
                edge_index=edge_index,
                sp_edge_index=torch.tensor(g['sp_edge_index'], dtype=torch.long),
                sp_value=torch.tensor(g['sp_value'], dtype=torch.float32),
                sp_edge_rel=torch.tensor(g['sp_edge_rel'], dtype=torch.long),
            )

    def _build_batch(self, drug_ids: torch.Tensor) -> Batch:
        device = next(self.parameters()).device
        ids = drug_ids.cpu().tolist()
        data_list = [self._cache[did] for did in ids]
        return Batch.from_data_list(data_list).to(device)

    def forward(self, drug_ids: torch.Tensor) -> torch.Tensor:
        from torch_geometric.utils import degree as pyg_degree
        batch = self._build_batch(drug_ids)
        x_norm = self.atom_norm(batch.x)
        x = self.atom_embed(x_norm)
        x_degree = pyg_degree(batch.edge_index[1], batch.x.size(0),
                              dtype=torch.long).clamp(max=self.max_degree)
        x = x + self.degree_encoder(x_degree)
        x = self.gt(x, batch.sp_edge_index, batch.sp_value, batch.sp_edge_rel)
        out = global_mean_pool(x, batch.batch)            # [B, hidden_dim]
        return self.dropout(out)

# ============================================================================
# RELATION / KG ENCODERS  (cells 08 TransE / 09 RGCN-sub / 10 GT-KG / 11 RGCN-whole)
# ============================================================================

class TransEEncoder(nn.Module):
    """Cell 08 (REL): TransE frozen embedding lookup on HetioNet KG.

    Source: MUFFIN (Chen et al., 2021) — KGE(pre-trained) 약물 표현.
    Fair-exp adaptation: DDI-free HetioNet에 사전학습한 TransE 임베딩을 frozen lookup
      (학습 중 갱신 X) 후 Linear+ReLU+Dropout 사영만 학습; drug_id == entity_id identity map
      (별도 drug2kgid 불필요). DDI-334: bio KG + train.txt typed로 재학습한 KGE 사용.
    """

    def __init__(self, hidden_dim: int = 128, num_entities: int = 34124,
                 embed_dim: int = 128,
                 dropout: float = 0.2, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_entities = num_entities
        self.embed_dim = embed_dim
        self.entity_embedding = nn.Embedding(num_entities, embed_dim)
        self.entity_embedding.weight.requires_grad = False  # frozen
        self.project = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Identity mapping: drug_id -> entity_id (drug 0-1709 = entity 0-1709)
        self._drug2kg_table: Optional[torch.Tensor] = None

    def set_features(self, drug2kgid: Optional[Dict[int, int]] = None,
                     num_drugs: int = 1710):
        """Build drug_id -> kg_id mapping.

        For HetioNet, drug_id == entity_id (identity), so drug2kgid can be None.
        If drug2kgid is provided, it is used as-is. Otherwise, identity mapping
        for drug ids 0..num_drugs-1.
        """
        if drug2kgid is not None:
            max_drug_id = max(drug2kgid.keys()) if drug2kgid else 0
            mapping = torch.zeros(max_drug_id + 1, dtype=torch.long)
            for did, kgid in drug2kgid.items():
                mapping[did] = kgid
        else:
            # Identity mapping: drug_id 0..num_drugs-1 -> entity_id 0..num_drugs-1
            mapping = torch.arange(num_drugs, dtype=torch.long)
        self._drug2kg_table = mapping

    def load_pretrained(self, embedding_path: str):
        """Load pre-trained TransE entity embeddings from .npy or .pt file."""
        if embedding_path.endswith('.npy'):
            embeddings = torch.from_numpy(np.load(embedding_path)).float()
        else:
            embeddings = torch.load(embedding_path, map_location='cpu')
            if isinstance(embeddings, dict):
                embeddings = embeddings.get('entity_embedding', embeddings)
        assert embeddings.shape[0] == self.num_entities, (
            f"Embedding rows {embeddings.shape[0]} != num_entities {self.num_entities}")
        self.entity_embedding = nn.Embedding.from_pretrained(embeddings, freeze=True)
        self.embed_dim = embeddings.shape[1]

    def forward(self, drug_ids: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        if self._drug2kg_table is None:
            # Fallback: identity
            kg_ids = drug_ids.long()
        else:
            if self._drug2kg_table.device != device:
                self._drug2kg_table = self._drug2kg_table.to(device)
            kg_ids = self._drug2kg_table[drug_ids.long()]  # [B]
        emb = self.entity_embedding(kg_ids)
        return self.project(emb)


# ── R-GCN on pair-wise subgraph ──

class SubgraphExtractor:
    """
    Extract k-hop enclosing subgraphs around drug pairs from KG.
    Following SumGNN: BFS k-hop, intersection, max_nodes_per_hop sampling.
    Pre-computes and caches subgraphs.

    For HetioNet:
      - num_entities = 34124
      - num_relations = 23 (KG.txt has rel 86-108, caller passes offset-adjusted 0-22)
      - hop = 2, max_nodes_per_hop = 200
    """

    def __init__(self, triples: np.ndarray, num_entities: int, num_relations: int,
                 hop: int = 2, max_nodes_per_hop: int = 200, cache_dir: str = None):
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.hop = hop
        self.max_nodes_per_hop = max_nodes_per_hop

        # Build adjacency list (undirected, relation-typed)
        # adj[node] = [(neighbor, relation_id), ...]
        import pickle, hashlib
        if cache_dir is None:
            cache_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'subgraph_cache')
        cache_dir = os.path.abspath(cache_dir)
        adj_hash = hashlib.md5(f"{len(triples)}_{num_entities}_{num_relations}".encode()).hexdigest()[:8]
        adj_cache = os.path.join(cache_dir, f'adj_list_{adj_hash}.pkl')

        if os.path.exists(adj_cache):
            print(f"  Loading cached adjacency list from {adj_cache}")
            with open(adj_cache, 'rb') as f:
                raw = pickle.load(f)
            # Wrap in defaultdict so missing nodes return [] without KeyError
            self.adj = defaultdict(list, raw)
        else:
            print(f"  Building adjacency list ({len(triples)} triples)...")
            self.adj = defaultdict(list)
            for h, t, r in triples:  # triples format: (head, tail, relation)
                self.adj[h].append((t, r))
                self.adj[t].append((h, r + num_relations))  # inverse relation
            # Save cache
            os.makedirs(cache_dir, exist_ok=True)
            with open(adj_cache, 'wb') as f:
                pickle.dump(dict(self.adj), f)
            print(f"  Adjacency list cached to {adj_cache}")
        self.total_relations = num_relations * 2  # original + inverse

        # Build numpy CSR adjacency for fast vectorized BFS + edge extraction
        max_node_id = max(self.adj.keys()) if self.adj else 0
        _csr_num_nodes = max_node_id + 1
        _all_src, _all_dst, _all_rel = [], [], []
        for s, nbrs in self.adj.items():
            for (t, r) in nbrs:
                _all_src.append(s)
                _all_dst.append(t)
                _all_rel.append(r)
        if _all_src:
            src_arr = np.array(_all_src, dtype=np.int64)
            dst_arr = np.array(_all_dst, dtype=np.int64)
            rel_arr = np.array(_all_rel, dtype=np.int64)
            sort_idx = np.argsort(src_arr, kind='stable')
            src_sorted = src_arr[sort_idx]
            self.csr_dst = dst_arr[sort_idx]
            self.csr_rel = rel_arr[sort_idx]
            degree = np.bincount(src_sorted, minlength=_csr_num_nodes).astype(np.int64)
            self.csr_ptr = np.zeros(_csr_num_nodes + 1, dtype=np.int64)
            np.cumsum(degree, out=self.csr_ptr[1:])
        else:
            self.csr_dst = np.array([], dtype=np.int64)
            self.csr_rel = np.array([], dtype=np.int64)
            self.csr_ptr = np.zeros(_csr_num_nodes + 1, dtype=np.int64)
        self.csr_num_nodes = _csr_num_nodes
        # Reusable node_map_arr for global→local index mapping (filled per extract call)
        self._node_map_arr = np.full(_csr_num_nodes, -1, dtype=np.int64)

        # Cache (can be pre-loaded from disk via load_cache())
        self._cache: Dict[Tuple[int, int], Data] = {}

    def load_cache(self, cache_path: str):
        """Load pre-computed subgraphs from disk."""
        import pickle
        with open(cache_path, 'rb') as f:
            self._cache = pickle.load(f)
        print(f"  Loaded {len(self._cache)} cached subgraphs from {cache_path}")

    def save_cache(self, cache_path: str):
        """Persist in-memory subgraph cache to disk."""
        import pickle, os
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(self._cache, f)
        print(f"  Saved {len(self._cache)} subgraphs to {cache_path}")

    def precompute_all(self, pairs: List[Tuple[int, int]], verbose: bool = True,
                       transform_fn=None) -> None:
        """Precompute subgraphs (and optionally tiger-transform) for all pairs.

        Args:
            pairs: list of (kg_id1, kg_id2) drug pairs
            transform_fn: optional callable (e.g. GTSubgraphEncoder._to_tiger_inputs).
                If provided, called on each subgraph Data — result stored as
                data._tiger_inputs so workers return instantly on next access.

        Call before DataLoader creation so workers inherit cache via CoW fork.
        """
        missing = [(h, t) for h, t in pairs if (h, t) not in self._cache]
        if not missing:
            if verbose:
                print(f"  All {len(pairs)} subgraphs already cached.")
            # Still run transform on any un-transformed cached entries
            if transform_fn is not None:
                for h, t in pairs:
                    d = self._cache.get((h, t))
                    if d is not None and not hasattr(d, '_tiger_inputs'):
                        transform_fn(d)
            return
        if verbose:
            label = "+tiger" if transform_fn is not None else ""
            print(f"  Precomputing {len(missing)} subgraphs{label}... ", end='', flush=True)
        import time
        t0 = time.time()
        for h, t in missing:
            data = self.extract(h, t)
            if transform_fn is not None:
                transform_fn(data)  # side effect: sets data._tiger_inputs
        elapsed = time.time() - t0
        if verbose:
            print(f"done in {elapsed:.0f}s ({elapsed/len(missing)*1000:.1f}ms/pair)")

    def extract(self, head_kg_id: int, tail_kg_id: int) -> Data:
        """Extract enclosing k-hop subgraph for a drug pair."""
        key = (head_kg_id, tail_kg_id)
        if key in self._cache:
            return self._cache[key]

        # BFS k-hop neighborhoods
        head_neighbors = self._bfs_khop(head_kg_id)
        tail_neighbors = self._bfs_khop(tail_kg_id)

        # Enclosing subgraph: intersection
        all_head = set()
        for level in head_neighbors:
            all_head |= level
        all_tail = set()
        for level in tail_neighbors:
            all_tail |= level

        subgraph_nodes = all_head & all_tail
        subgraph_nodes.add(head_kg_id)
        subgraph_nodes.add(tail_kg_id)
        subgraph_nodes = sorted(subgraph_nodes)

        # Build global→local index map using numpy array (reused buffer, reset after)
        sg_arr = np.array(subgraph_nodes, dtype=np.int64)
        N = len(sg_arr)
        # local 0 = head, 1 = tail, 2..N-1 = others (sorted)
        local_order = [head_kg_id, tail_kg_id] + [n for n in subgraph_nodes
                                                    if n not in (head_kg_id, tail_kg_id)]
        local_arr = np.array(local_order, dtype=np.int64)
        self._node_map_arr[local_arr] = np.arange(N, dtype=np.int64)

        global_ids = torch.tensor(local_arr, dtype=torch.long)

        # Vectorized edge extraction via numpy CSR
        # Gather all adj entries for subgraph nodes in one pass
        seg_len = np.array([int(self.csr_ptr[n+1] - self.csr_ptr[n]) for n in sg_arr],
                           dtype=np.int64)
        total_adj = seg_len.sum()
        if total_adj > 0:
            flat_idx = np.concatenate([np.arange(int(self.csr_ptr[n]),
                                                   int(self.csr_ptr[n+1]), dtype=np.int64)
                                        for n in sg_arr])
            dst_all = self.csr_dst[flat_idx]
            rel_all = self.csr_rel[flat_idx]
            src_all = np.repeat(sg_arr, seg_len)

            # Keep only dst that is in subgraph
            dst_local = self._node_map_arr[np.clip(dst_all, 0, self.csr_num_nodes - 1)]
            in_sg = dst_local >= 0
            # Remove head↔tail direct edges (leakage prevention)
            is_ht = ((src_all == head_kg_id) & (dst_all == tail_kg_id)) | \
                    ((src_all == tail_kg_id) & (dst_all == head_kg_id))
            mask = in_sg & ~is_ht

            src_local = self._node_map_arr[src_all[mask]]
            dst_local = dst_local[mask]
            rel_local = rel_all[mask]
        else:
            src_local = np.array([], dtype=np.int64)
            dst_local = np.array([], dtype=np.int64)
            rel_local = np.array([], dtype=np.int64)

        # Reset node_map_arr slots to -1 (reuse buffer without full zeroing)
        self._node_map_arr[local_arr] = -1

        if len(src_local) == 0:
            # Fallback: self-loops so GNN has at least one message
            src_local = np.array([0, 1], dtype=np.int64)
            dst_local = np.array([0, 1], dtype=np.int64)
            rel_local = np.array([0, 0], dtype=np.int64)

        data = Data(
            num_nodes=N,
            edge_index=torch.tensor(np.stack([src_local, dst_local], axis=0), dtype=torch.long),
            edge_type=torch.tensor(rel_local, dtype=torch.long),
            global_ids=global_ids,
        )

        self._cache[key] = data
        return data

    def _bfs_khop(self, start: int) -> List[set]:
        """BFS returning level sets for k hops — numpy CSR + bool-array version.

        Uses a boolean visited mask over the full entity range to avoid the
        O(N log N) np.unique bottleneck (~100x faster than Python BFS for
        high-degree KG nodes where hop-2 frontiers reach 60K+ candidates).
        """
        max_id = self.csr_num_nodes
        visited_bool = np.zeros(max_id, dtype=bool)
        visited_bool[start] = True
        frontier = np.array([start], dtype=np.int64)
        levels = []
        for _ in range(self.hop):
            # Gather all neighbor IDs from CSR
            slices = [self.csr_dst[self.csr_ptr[n]:self.csr_ptr[n + 1]]
                      for n in frontier if n < max_id]
            if slices:
                cat = np.concatenate(slices)
                # Dedup and filter visited in O(max_id) using bool array
                seen = np.zeros(max_id, dtype=bool)
                seen[cat] = True
                seen &= ~visited_bool
                candidates = np.where(seen)[0]
            else:
                candidates = np.array([], dtype=np.int64)

            if len(candidates) > self.max_nodes_per_hop:
                idx = np.random.choice(len(candidates), self.max_nodes_per_hop, replace=False)
                candidates = candidates[idx]

            visited_bool[candidates] = True
            frontier = candidates
            levels.append(set(candidates.tolist()))
        return levels


class RGCNSubgraphEncoder(nn.Module):
    """Cell 09 (REL): R-GCN on pair-wise k=2 hop enclosing subgraph (KG).

    Source: SumGNN (Yu et al., 2021) — KG subgraph + R-GCN.
    Fair-exp adaptation: SumGNN의 DRNL(double-radius node labeling) 제거 — Cell 10(GT)과
      노드 초기화를 TransE frozen init으로 통일해 같은 subgraph 위 아키텍처(R-GCN vs GT)만
      통제 비교하기 위함; k=2 hop enclosing subgraph만 추출(pair별 캐시).

    HetioNet defaults:
      - num_entities = 34124
      - num_relations = 23 (KG.txt offset-adjusted; x2 inverse applied internally)
      - num_bases = 4 (SumGNN R/B)
      - num_layers = 2, edge_dropout = 0.4

    input_dim: KGE embedding dim (e.g. 304 for TransE). First RGCNConv projects
    input_dim -> hidden_dim. If None, input_dim = hidden_dim (identity).
    With KGE frozen init (use_transe_init=True, default for Cell 09).
    """

    def __init__(self, hidden_dim: int = 128, num_entities: int = 34124,
                 num_relations: int = 23,  # HetioNet non-DDI relations; x2 inverse applied internally
                 num_bases: int = 4, num_layers: int = 2,
                 dropout: float = 0.2, edge_dropout: float = 0.4,
                 use_transe_init: bool = True, input_dim: int = None, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_entities = num_entities
        self.num_relations = num_relations * 2  # include inverse
        self.num_layers = num_layers
        self.use_transe_init = use_transe_init
        self.edge_dropout = edge_dropout
        self.input_dim = input_dim or hidden_dim

        self.node_embedding = nn.Embedding(num_entities, self.input_dim)
        self.node_embedding.weight.requires_grad = not use_transe_init

        # R-GCN layers: first projects input_dim -> hidden_dim, rest hidden -> hidden
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(num_layers):
            in_ch = self.input_dim if i == 0 else hidden_dim
            self.convs.append(RGCNConv(
                in_channels=in_ch,
                out_channels=hidden_dim,
                num_relations=self.num_relations,
                num_bases=num_bases,
            ))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.dropout = dropout

        # External data
        self._drug2kg_table: Optional[torch.Tensor] = None
        self.subgraph_extractor: Optional[SubgraphExtractor] = None

    def set_features(self, drug2kgid: Optional[Dict[int, int]] = None,
                     num_drugs: int = 1710):
        """Set drug_id -> kg_id mapping.

        For HetioNet, drug_id == entity_id (identity), so drug2kgid can be None.
        """
        if drug2kgid is not None:
            max_drug_id = max(drug2kgid.keys()) if drug2kgid else 0
            mapping = torch.zeros(max_drug_id + 1, dtype=torch.long)
            for did, kgid in drug2kgid.items():
                mapping[did] = kgid
        else:
            mapping = torch.arange(num_drugs, dtype=torch.long)
        self._drug2kg_table = mapping

    def set_kg_data(self, triples: np.ndarray, num_entities: int, num_relations: int,
                    cache_dir: str = None):
        """Initialize subgraph extractor from KG triples. num_relations = original count."""
        self.num_entities = num_entities
        self.subgraph_extractor = SubgraphExtractor(
            triples, num_entities, num_relations,
            hop=2, max_nodes_per_hop=200, cache_dir=cache_dir,
        )

    def load_transe_init(self, embeddings: torch.Tensor):
        """Load KGE embeddings as frozen node features (any dim)."""
        self.node_embedding = nn.Embedding.from_pretrained(embeddings, freeze=True)

    def _extract_batch_subgraphs(self, drug1_ids: torch.Tensor,
                                  drug2_ids: torch.Tensor) -> Batch:
        """Extract and batch subgraphs for all drug pairs."""
        if self._drug2kg_table is not None:
            kg1_ids = self._drug2kg_table[drug1_ids.cpu().long()].tolist()
            kg2_ids = self._drug2kg_table[drug2_ids.cpu().long()].tolist()
        else:
            kg1_ids = drug1_ids.cpu().tolist()
            kg2_ids = drug2_ids.cpu().tolist()
        data_list = [self.subgraph_extractor.extract(k1, k2)
                     for k1, k2 in zip(kg1_ids, kg2_ids)]
        return Batch.from_data_list(data_list)

    def forward_pair(self, drug1_ids: torch.Tensor,
                     drug2_ids: torch.Tensor,
                     rel_subgraph_batch: Batch = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode drug pairs via R-GCN on their enclosing subgraphs.
        Returns per-drug contextualized embeddings: (h_d1[B,H], h_d2[B,H])
        """
        device = next(self.parameters()).device

        if rel_subgraph_batch is not None:
            batch = rel_subgraph_batch.to(device)
        else:
            batch = self._extract_batch_subgraphs(drug1_ids, drug2_ids)
            batch = batch.to(device)

        # Node features: KGE lookup
        x = self.node_embedding(batch.global_ids)  # [N, hidden_dim]

        edge_index = batch.edge_index
        edge_type = batch.edge_type

        # Edge dropout during training
        if self.training and self.edge_dropout > 0:
            mask = torch.rand(edge_type.size(0), device=device) > self.edge_dropout
            edge_index = edge_index[:, mask]
            edge_type = edge_type[mask]

        # R-GCN message passing
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_type)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Per-drug readout: node 0 = drug1, node 1 = drug2
        starts = batch.ptr[:-1]  # [B]
        h_d1 = x[starts]        # [B, H]
        h_d2 = x[starts + 1]    # [B, H]
        return h_d1, h_d2

    def forward(self, drug_ids: torch.Tensor) -> torch.Tensor:
        """Single drug encoding fallback (embedding lookup only, no GNN).
        DDIModel should always use forward_pair() for R-GCN encoders."""
        import warnings
        warnings.warn(
            "RGCNSubgraphEncoder.forward() returns raw embeddings without GNN. "
            "Use forward_pair() for proper subgraph-contextualized encoding.",
            stacklevel=2,
        )
        device = next(self.parameters()).device
        if self._drug2kg_table is not None:
            if self._drug2kg_table.device != device:
                self._drug2kg_table = self._drug2kg_table.to(device)
            kg_ids = self._drug2kg_table[drug_ids.long()]
        else:
            kg_ids = drug_ids.long()
        return self.node_embedding(kg_ids)


# ══════════════════════════════════════════════════════════════
# Cell 11: R-GCN on whole DDI graph (Decagon-style)
#
# Ref: Decagon (Zitnik et al., 2018, Bioinformatics).
# Whole-graph propagation over DDI train graph (1710 drug nodes, 86×2 relations).
# TransE init (drug rows 0-1709 of HetioNet entity embedding) — fine-tunable.
# ══════════════════════════════════════════════════════════════

class RGCNWholeDDIEncoder(nn.Module):
    """Cell 11 (REL): R-GCN on whole DDI train graph (drug nodes only).

    Source: Decagon (Zitnik et al., 2018) — multi-relational GCN on DDI graph.
    Fair-exp adaptation: TransE 임베딩을 frozen init으로 사용(Cell 08/09/10과 통일 —
      모든 REL이 고정 TransE 노드 입력 위에서 인코더만 비교); KG 없이 DDI train 그래프
      (drug 노드만) R-GCN 전파; Cell 09/10(pair subgraph)과 달리 전체 그래프를 forward당
      1회 전파 후 index lookup (rel_is_pairwise=False).
    Defaults: num_drugs=1710, num_relations=2*n_types(inverse 포함),
      num_bases=30, num_layers=2, edge_dropout=0.4.
    """

    def __init__(self, hidden_dim: int = 128, num_drugs: int = 1710,
                 num_relations: int = 172,
                 num_bases: int = 30, num_layers: int = 2,
                 dropout: float = 0.2, edge_dropout: float = 0.4,
                 use_transe_init: bool = True, input_dim: int = None, **kwargs):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_drugs = num_drugs
        self.num_relations = num_relations
        self.num_layers = num_layers
        self.edge_dropout = edge_dropout
        self.input_dim = input_dim or hidden_dim

        self.node_embedding = nn.Embedding(num_drugs, self.input_dim)
        if not use_transe_init:
            nn.init.xavier_uniform_(self.node_embedding.weight)
        # TransE init이면 frozen (Cell 08/09/10과 통일 — 모든 REL이 고정 TransE 노드 입력).
        # random init(use_transe_init=False)일 때만 학습.
        self.node_embedding.weight.requires_grad = not use_transe_init

        # First conv projects input_dim -> hidden_dim, rest are hidden -> hidden
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_ch = self.input_dim if i == 0 else hidden_dim
            self.convs.append(RGCNConv(
                in_channels=in_ch,
                out_channels=hidden_dim,
                num_relations=num_relations,
                num_bases=num_bases,
            ))
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

        self.register_buffer('_edge_index', torch.zeros((2, 0), dtype=torch.long),
                             persistent=False)
        self.register_buffer('_edge_type', torch.zeros((0,), dtype=torch.long),
                             persistent=False)

    def set_transe_init(self, transe_drug_embeddings: torch.Tensor):
        """Init node embeddings from KGE drug rows (any dim — must match input_dim)."""
        with torch.no_grad():
            self.node_embedding.weight.copy_(transe_drug_embeddings.float())

    def set_ddi_graph(self, edge_index: torch.Tensor, edge_type: torch.Tensor):
        """Register DDI train graph (call once before training)."""
        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError("edge_index must be [2, num_edges]")
        if edge_type.size(0) != edge_index.size(1):
            raise ValueError("edge_type length must equal num_edges")
        self._edge_index = edge_index.long()
        self._edge_type = edge_type.long()

    def _propagate_whole_graph(self) -> torch.Tensor:
        device = next(self.parameters()).device
        x = self.node_embedding.weight                  # [num_drugs, H]
        # Move edge tensors to device once (avoid per-batch CPU→GPU copy)
        if self._edge_index.device != device:
            self._edge_index = self._edge_index.to(device)
            self._edge_type = self._edge_type.to(device)
        edge_index = self._edge_index
        edge_type = self._edge_type
        if self.training and self.edge_dropout > 0 and edge_index.size(1) > 0:
            keep = torch.rand(edge_index.size(1), device=device) > self.edge_dropout
            edge_index = edge_index[:, keep]
            edge_type = edge_type[keep]
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_type)
            x = norm(x)
            x = F.relu(x)
            x = self.dropout(x)
        return x                                        # [num_drugs, H]

    def forward(self, drug_ids: torch.Tensor) -> torch.Tensor:
        all_emb = self._propagate_whole_graph()
        return all_emb[drug_ids.long()]


# ══════════════════════════════════════════════════════════════
# Cell 10: Graph Transformer (TIGER-style) on KG subgraph
#
# Reuses _TigerStyleGT from mol_encoders.py (same class, different num_rel).
# For HetioNet: num_relations=23, so num_rel_total = 23 + max_distance + 1 = 32
# ══════════════════════════════════════════════════════════════

class GraphTransformerKGEncoder(nn.Module):
    """Cell 10 (REL): TIGER-style GT on KG subgraph (Cell 03 GT layer reused).

    Source: TIGER (Su et al., 2024) — Graph Transformer (Cell 03과 동일 layer).
    Fair-exp adaptation: Cell 03의 TIGER GT layer(동일 attention)를 KG subgraph에 재사용
      → Cell 09(R-GCN)와 같은 subgraph+TransE init 위에서 GT vs R-GCN만 통제 비교;
      TIGER GT attention은 repo의 linear/Performer 공식 사용 (paper eq(2) softmax 아님);
      TransE frozen init; subgraph는 pair별 캐시.

    HetioNet defaults:
      - num_entities = 34124
      - num_relations = 23 (original; indirect slots = 23..23+max_distance)
      - hidden_dim = 128, num_layers = 2, num_heads = 4
      - max_distance = 8, max_degree = 50
      - use_transe_init = True (TransE frozen init)

    Subgraph -> sp_edge_* conversion via _to_tiger_inputs() (cached per pair).
    """

    def __init__(self, num_entities: int = 34124,
                 num_relations: int = 23,
                 hidden_dim: int = 128,
                 num_layers: int = 2, num_heads: int = 4,
                 max_distance: int = 8, max_degree: int = 50,
                 dropout: float = 0.2,
                 use_transe_init: bool = True, input_dim: int = None, **kwargs):
        super().__init__()
        # _TigerStyleGT는 동일 모듈(위 MOL 섹션)에 정의됨 — 별도 import 불필요
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.max_distance = max_distance
        self.max_degree = max_degree
        self.input_dim = input_dim or hidden_dim
        # rel embedding spans: 0..num_relations-1 = direct, num_relations+d = indirect
        num_rel_total = num_relations + max_distance + 1

        # KG entity embedding (KGE init or random); input_dim may differ from hidden_dim
        self.entity_embedding = nn.Embedding(num_entities, self.input_dim)
        self.entity_embedding.weight.requires_grad = not use_transe_init
        if not use_transe_init:
            nn.init.xavier_uniform_(self.entity_embedding.weight)
        # Project input_dim -> hidden_dim before GT (identity if dims match)
        if self.input_dim != hidden_dim:
            self.input_proj = nn.Linear(self.input_dim, hidden_dim)
        else:
            self.input_proj = None
        # Degree encoder (TIGER NodeFeatures convention)
        self.degree_encoder = nn.Embedding(max_degree + 1, hidden_dim, padding_idx=0)
        # GT stack — same _TigerStyleGT class as Cell 03 (different num_rel only)
        self.gt = _TigerStyleGT(hidden_dim, num_layers, num_heads,
                                num_rel_total, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        # External
        self._drug2kg_table: Optional[torch.Tensor] = None
        self.subgraph_extractor: Optional[SubgraphExtractor] = None

    def set_features(self, drug2kgid: Optional[Dict[int, int]] = None,
                     num_drugs: int = 1710):
        """Set drug_id -> kg_id mapping.

        For HetioNet, drug_id == entity_id (identity), so drug2kgid can be None.
        """
        if drug2kgid is not None:
            max_drug_id = max(drug2kgid.keys()) if drug2kgid else 0
            mapping = torch.zeros(max_drug_id + 1, dtype=torch.long)
            for did, kgid in drug2kgid.items():
                mapping[did] = kgid
        else:
            mapping = torch.arange(num_drugs, dtype=torch.long)
        self._drug2kg_table = mapping

    def set_transe_init(self, transe_entity_emb: torch.Tensor):
        """Initialize entity embeddings from KGE (frozen, any dim)."""
        with torch.no_grad():
            self.entity_embedding.weight.copy_(transe_entity_emb.float())

    def set_kg_data(self, triples: np.ndarray, num_entities: int,
                    num_relations: int, **subgraph_kwargs):
        """Initialize subgraph extractor.

        num_relations here is the *original* (non-doubled) count; the
        extractor internally doubles for inverse edges.
        """
        self.subgraph_extractor = SubgraphExtractor(
            triples, num_entities, num_relations, **subgraph_kwargs)

    @staticmethod
    def _bfs_distances(num_nodes: int, edge_index: np.ndarray,
                       max_distance: int) -> np.ndarray:
        """All-pairs shortest path distances (unweighted, undirected).

        Uses scipy sparse graph BFS (C implementation) — ~100x faster than
        pure-Python BFS for N~800 subgraphs.
        Returns int32 array [N, N]; unreachable pairs = max_distance + 1.
        """
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import shortest_path
        INF = max_distance + 1
        if edge_index.size == 0 or num_nodes == 0:
            d = np.full((num_nodes, num_nodes), INF, dtype=np.int32)
            np.fill_diagonal(d, 0)
            return d
        row = np.concatenate([edge_index[0], edge_index[1]])
        col = np.concatenate([edge_index[1], edge_index[0]])
        data = np.ones(len(row), dtype=np.float32)
        adj_sp = csr_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
        # shortest_path returns float64 inf for unreachable; clip to INF
        sp = shortest_path(adj_sp, method='D', directed=False, unweighted=True)
        sp = np.where(np.isinf(sp) | (sp > max_distance), INF, sp)
        return sp.astype(np.int32)

    def _to_tiger_inputs(self, data) -> 'Data':
        """Convert a Cell 09 subgraph Data into the tensors _TigerStyleGT expects.
        Cached on the subgraph object via data._tiger_inputs.

        Direct-edge precedence: if a pair (i, j) has a direct relation in the
        subgraph, that relation_id wins over the distance-based rel slot.
        Vectorized with numpy to avoid O(N^2) Python loop overhead.
        """
        if hasattr(data, '_tiger_inputs') and data._tiger_inputs is not None:
            return data._tiger_inputs
        N = int(data.global_ids.size(0))
        edge_index_np = data.edge_index.cpu().numpy()
        edge_type_np = data.edge_type.cpu().numpy()
        if edge_index_np.size:
            edge_index_np = np.clip(edge_index_np, 0, N - 1)

        # Shortest path distances within the subgraph — shape [N, N]
        dist = self._bfs_distances(N, edge_index_np, self.max_distance)

        # Vectorized: find all (i, j) with dist <= max_distance
        i_idx, j_idx = np.where(dist <= self.max_distance)
        d_vals = dist[i_idx, j_idx].astype(np.int64)

        # Default to distance-based relation slot
        rels = (self.num_relations + d_vals).astype(np.int64)
        vals = d_vals.astype(np.float32)

        # Build dense direct-relation matrix once (N×N, -1 = no direct edge)
        # E is small (few hundred), so this loop is cheap
        direct_mat = np.full((N, N), -1, dtype=np.int64)
        if edge_index_np.size:
            u_arr = edge_index_np[0]
            v_arr = edge_index_np[1]
            r_arr = (edge_type_np % self.num_relations).astype(np.int64)
            for k in range(len(u_arr)):
                if direct_mat[u_arr[k], v_arr[k]] < 0:  # keep first
                    direct_mat[u_arr[k], v_arr[k]] = r_arr[k]

        # Override distance-based entries with direct relation where present
        direct_lookup = direct_mat[i_idx, j_idx]
        has_direct = direct_lookup >= 0
        rels[has_direct] = direct_lookup[has_direct]
        vals[has_direct] = 1.0

        sp_edge_index = torch.tensor(np.stack([i_idx, j_idx], axis=0), dtype=torch.long)
        sp_edge_rel = torch.tensor(rels, dtype=torch.long)
        sp_value = torch.tensor(vals, dtype=torch.float32)

        out = Data(
            num_nodes=N,
            global_ids=data.global_ids,
            edge_index=data.edge_index,        # direct edges (for degree calc)
            edge_type=data.edge_type,
            sp_edge_index=sp_edge_index,
            sp_edge_rel=sp_edge_rel,
            sp_value=sp_value,
        )
        data._tiger_inputs = out
        return out

    def _extract_batch_subgraphs(self, drug1_ids: torch.Tensor,
                                  drug2_ids: torch.Tensor) -> 'Batch':
        if self._drug2kg_table is not None:
            kg1 = self._drug2kg_table[drug1_ids.cpu().long()].tolist()
            kg2 = self._drug2kg_table[drug2_ids.cpu().long()].tolist()
        else:
            kg1 = drug1_ids.cpu().tolist()
            kg2 = drug2_ids.cpu().tolist()
        data_list = []
        for k1, k2 in zip(kg1, kg2):
            sg = self.subgraph_extractor.extract(k1, k2)
            data_list.append(self._to_tiger_inputs(sg))
        return Batch.from_data_list(data_list)

    def forward_pair(self, drug1_ids: torch.Tensor,
                     drug2_ids: torch.Tensor,
                     rel_subgraph_batch: 'Batch' = None
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a drug pair via TIGER GT on its enclosing KG subgraph.

        Returns (h_d1, h_d2) of shape [B, hidden_dim].
        """
        from torch_geometric.utils import degree as pyg_degree

        device = next(self.parameters()).device
        if rel_subgraph_batch is not None:
            # collate_fn applies _to_tiger_inputs when subgraph_transform is set,
            # so the batch already has sp_edge_*. Use it directly.
            batch = rel_subgraph_batch.to(device)
        else:
            batch = self._extract_batch_subgraphs(drug1_ids, drug2_ids).to(device)

        # Node features: KGE entity emb (projected if needed) + degree
        x = self.entity_embedding(batch.global_ids)
        if self.input_proj is not None:
            x = self.input_proj(x)
        num_nodes = batch.global_ids.size(0)
        deg = pyg_degree(batch.edge_index[1], num_nodes,
                         dtype=torch.long).clamp(max=self.max_degree)
        x = x + self.degree_encoder(deg)

        # GT stack (TIGER 4-th root + linear normalization + spatial bias)
        x = self.gt(x, batch.sp_edge_index, batch.sp_value, batch.sp_edge_rel)
        x = self.dropout(x)

        # Per-pair head/tail embeddings (node 0 = drug1, node 1 = drug2)
        starts = batch.ptr[:-1]                          # [B]
        h_d1 = x[starts]
        h_d2 = x[starts + 1]
        return h_d1, h_d2

    def forward(self, drug_ids: torch.Tensor) -> torch.Tensor:
        """Single-drug fallback (encoder API contract).

        Returns plain TransE entity embedding (no GT).
        DDIModel should call forward_pair() for proper subgraph-contextualized encoding.
        """
        import warnings
        warnings.warn(
            "GraphTransformerKGEncoder.forward() returns raw TransE embeddings "
            "without GT. Use forward_pair() for subgraph-contextualized output.",
            stacklevel=2,
        )
        device = next(self.parameters()).device
        if self._drug2kg_table is not None:
            if self._drug2kg_table.device != device:
                self._drug2kg_table = self._drug2kg_table.to(device)
            kg_ids = self._drug2kg_table[drug_ids.long()]
        else:
            kg_ids = drug_ids.long()
        return self.entity_embedding(kg_ids)

# NOTE: Cell 12 (Interaction-Jaccard)는 frozen 벡터->MLP 구조라
#       models/frozen_encoder.py로 통합. rel_encoders는 KG/그래프 인코더만 보유.
