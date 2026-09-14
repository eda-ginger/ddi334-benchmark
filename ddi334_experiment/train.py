"""
V5 DDI experiment trainer — Cells 01-14 (single-source, Molecule/Relation/Document).

Usage:
  python train.py --cell 01 --gpu 0 --seed 42
  python train.py --cell 09 --gpu 1 --seed 0

Cells:
  MOL:
  01  GAT + SAGPool on mol graph (55-dim atom features)
  02  SchNet on 3D mol graph (all 1710, drug 1050 = MANUAL_OCTAHEDRAL)
  03  Graph Transformer mol (TIGER-style, 67-dim atom features)
  04  MLP on Morgan FP (ECFP4, 1024-dim)
  05  FP Jaccard + train-only PCA (DDIMDL, leakage-fixed)
  06  ChemBERTa frozen [CLS] (384->128)
  07  1D CNN on SMILES (char-level)
  REL:
  08  TransE frozen + Linear proj (HetioNet, 34124 entities)
  09  R-GCN on HetioNet subgraph + TransE frozen init
  10  Graph Transformer on HetioNet subgraph + TransE frozen init
  11  R-GCN on whole DDI graph + TransE fine-tunable init (Decagon-style)
  12  Bio interaction profile + train-only PCA (DDIMDL, leakage-fixed)
  DOC:
  13  BioBERT frozen [CLS] description (768->128)
  14  BioBERT frozen [CLS] drug name (768->128)

Evaluates S0/S1/S2 splits in one training run.
"""
import os
import sys
import argparse
import json
import csv
import time
import subprocess
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.encoders import (
    GATEncoder, MorganFPEncoder, SchNetEncoder,
    GraphTransformerMolEncoder, SMILESCNNEncoder,
    ChemBERTaEncoder, BioBERTFrozenEncoder,
    TransEEncoder, RGCNSubgraphEncoder, GraphTransformerKGEncoder, RGCNWholeDDIEncoder,
)
from models.ddi_model import DDIModel
from dataio.dataset import build_dataloaders, build_collate_fn


# ── Paths ──
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
V5_DIR = os.path.dirname(EXPERIMENT_DIR)
DATA_DIR = os.path.join(V5_DIR, 'original_repo', 'DDI_Ben', 'DDI_Ben', 'data', 'drugbank_cluster')
KGE_DIR = os.path.join(V5_DIR, 'kge_hetionet')
KG_DATA_DIR = os.path.join(KGE_DIR, 'data')
# KGE type used for Cells 08-11. Override via --kge_type {transe,rotate,distmult,complex,transh}
KGE_TYPE = os.environ.get('KGE_TYPE', 'transe')

# All 1710 DDI-Bench drugs are used (no exclusion).
# Drug 1050 (Fe-CN complex) uses MANUAL_OCTAHEDRAL 3D coords in schnet_coords.pt.

# HetioNet entity / relation counts
NUM_HETIONET_ENTITIES = 34124
# Full HetioNet: rel 0-85 (DDI, from train.txt) + rel 86-108 (non-DDI, from KG.txt) = 109
NUM_HETIONET_RELATIONS = 109
# DDI-only graph (Cell 11): rel 0-85 = 86 types
NUM_DDI_RELATIONS = 86

# ── Notifications ──
NTFY_TOPIC = 'Latex_project'

def ntfy(msg: str):
    try:
        subprocess.Popen(
            ['curl', '-s', '-d', msg, f'ntfy.sh/{NTFY_TOPIC}'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass


# ── Hyperparameters ── (DDI-Bench 원본 기반, dropout 추가 / patience 제거 / checkpoint=Macro F1)
HP = {
    'hidden_dim': 128,
    'lr': 3e-4,
    'weight_decay': 1e-5,
    'epochs': 200,
    'batch_size': 128,
    'dropout': 0.2,
}


def load_drug_smiles(data_root: str = None) -> dict:
    """Load drug SMILES from DDI-Bench node2id.json + DrugBank SMILES lookup.

    Returns dict: drug_id (int) -> SMILES (str).
    Loads from KG data node2id.json (drug DB IDs -> entity IDs) and
    a smiles.json file if available. Falls back to RDKit-computed SMILES.
    """
    smiles_path = os.path.join(KG_DATA_DIR, '..', 'smiles.json')
    if not os.path.exists(smiles_path):
        smiles_path = os.path.join(EXPERIMENT_DIR, 'data', 'drug_smiles.json')
    if os.path.exists(smiles_path):
        with open(smiles_path) as f:
            raw = json.load(f)
        # raw: {drug_id_str: smiles} or {drugbank_id: smiles}
        # If keyed by int string, use directly
        smiles_dict = {}
        for k, v in raw.items():
            try:
                smiles_dict[int(k)] = v
            except ValueError:
                pass
        if smiles_dict:
            return smiles_dict
    raise FileNotFoundError(
        f"SMILES file not found. Expected at: {smiles_path}\n"
        "Run precompute_drug_smiles.py first.")


def load_drug_fingerprints(smiles_dict: dict) -> dict:
    """Compute Morgan FP (ECFP4, 1024-bit) from SMILES dict.
    Returns dict: drug_id -> np.ndarray [1024].
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    fps = {}
    for did, smi in smiles_dict.items():
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
            fps[did] = np.array(fp, dtype=np.float32)
        else:
            fps[did] = np.zeros(1024, dtype=np.float32)
    return fps


def load_mol_graphs_55d(smiles_dict: dict) -> dict:
    """Build 55-dim mol graphs for Cell 01 GAT."""
    from models.encoders import mol_to_graph_55d
    return {did: mol_to_graph_55d(smi) for did, smi in smiles_dict.items()}


def load_mol_graphs_tiger(smiles_dict: dict) -> dict:
    """Build TIGER-style mol graphs for Cell 03 GT."""
    from models.encoders import mol_to_graph_tiger
    return {did: mol_to_graph_tiger(smi) for did, smi in smiles_dict.items()}


def load_schnet_coords() -> dict:
    """Load precomputed 3D coordinates for SchNet (Cell 02).
    Returns dict: drug_id (int) -> {'z': np.ndarray, 'pos': np.ndarray}.
    """
    coords_path = os.path.join(EXPERIMENT_DIR, 'data', 'precompute', 'schnet_coords.pt')
    if not os.path.exists(coords_path):
        raise FileNotFoundError(
            f"SchNet 3D coords not found: {coords_path}\n"
            "Run: precompute/scripts/schnet_coords.py")
    return torch.load(coords_path, map_location='cpu', weights_only=False)


def load_chemberta_embeddings() -> dict:
    """Load precomputed frozen ChemBERTa [CLS] embeddings (Cell 06).
    Returns dict: drug_id (int) -> torch.Tensor [384].
    File format: {'embeddings': {int: Tensor[384]}, 'hidden_dim': 384, 'model_name': ...}
    """
    emb_path = os.path.join(EXPERIMENT_DIR, 'data', 'precompute', 'chemberta.pt')
    if not os.path.exists(emb_path):
        raise FileNotFoundError(
            f"ChemBERTa embeddings not found: {emb_path}\n"
            "Run: precompute/scripts/chemberta_embed.py")
    payload = torch.load(emb_path, map_location='cpu', weights_only=False)
    return payload['embeddings']   # {int: Tensor[384]}


def load_biobert_embeddings(input_type: str = 'description') -> dict:
    """Load precomputed frozen BioBERT [CLS] embeddings (Cells 13/14).

    Args:
        input_type: 'description' (Cell 13) or 'drugname' (Cell 14)
    Returns dict: drug_id (int) -> torch.Tensor [768].
    File format: {'embeddings': {int: Tensor[768]}, ...}
    """
    fname = {
        'description': 'biobert_description.pt',
        'drugname':    'biobert_drugname.pt',
    }[input_type]
    emb_path = os.path.join(EXPERIMENT_DIR, 'data', 'precompute', fname)
    if not os.path.exists(emb_path):
        raise FileNotFoundError(
            f"BioBERT {input_type} embeddings not found: {emb_path}\n"
            f"Run: precompute/scripts/biobert_embed.py --input_type {input_type}")
    payload = torch.load(emb_path, map_location='cpu', weights_only=False)
    return payload['embeddings']   # {int: Tensor[768]}


def load_hetionet_triples() -> np.ndarray:
    """Load full HetioNet: KG.txt (non-DDI) + train.txt (DDI), original rel IDs preserved.

    KG.txt:    h t r  where r in [86..108]  (non-DDI biological edges)
    train.txt: h t r  where r in [0..85]    (DDI training pairs)
    Combined:  rel 0-108, 109 relation types, no offset.

    Returns np.ndarray [N_triples, 3] with columns (h, t, r).
    """
    triples = []
    for path in [os.path.join(KG_DATA_DIR, 'KG.txt'),
                 os.path.join(KG_DATA_DIR, 'train.txt')]:
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 3:
                    h, t, r = int(parts[0]), int(parts[1]), int(parts[2])
                    triples.append((h, t, r))   # no offset — keep 0-108
    return np.array(triples, dtype=np.int64)


def load_ddi_triples() -> np.ndarray:
    """Load DDI-only graph from train.txt (Cell 11, Decagon-style).

    train.txt: h t r  where r in [0..85]  (86 DDI relation types)
    Returns np.ndarray [N_triples, 3] with columns (h, t, r).
    """
    train_path = os.path.join(KG_DATA_DIR, 'train.txt')
    triples = []
    with open(train_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 3:
                h, t, r = int(parts[0]), int(parts[1]), int(parts[2])
                triples.append((h, t, r))
    return np.array(triples, dtype=np.int64)


def build_ddi_edge_index(train_path: str, num_drugs: int = 1710):
    """Build DDI whole-graph edge_index + edge_type from train.txt.

    Adds inverse edges: forward rel r, inverse rel r+86.
    Returns (edge_index [2, 2E], edge_type [2E]).
    """
    src, dst, typ = [], [], []
    with open(train_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 3:
                continue
            h, t, r = int(parts[0]), int(parts[1]), int(parts[2])
            if max(h, t) >= num_drugs:
                continue
            src.append(h); dst.append(t); typ.append(r)
            src.append(t); dst.append(h); typ.append(r + NUM_DDI_RELATIONS)
    return (torch.tensor([src, dst], dtype=torch.long),
            torch.tensor(typ, dtype=torch.long))


def load_pca_embeddings(kind: str) -> dict:
    """Load precomputed PCA embeddings for Cell 05 (fp) or Cell 12 (bio).

    kind: 'fp'  -> data/fp_pca.pt   (FP Jaccard PCA, Cell 05)
          'bio' -> data/bio_pca.pt  (biological profile PCA, Cell 12)
    Returns dict {int drug_id: Tensor[128]} for fp, Tensor[384] for bio.
    """
    fname = 'fp_pca.pt' if kind == 'fp' else 'bio_pca.pt'
    path = os.path.join(EXPERIMENT_DIR, 'data', 'precompute', fname)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"PCA embeddings not found: {path}\n"
            f"Run precompute/scripts/{'fp' if kind == 'fp' else 'bio'}_pca.py first.")
    payload = torch.load(path, map_location='cpu', weights_only=False)
    return payload['embeddings']


def load_kge_embeddings(kge_type: str = None) -> torch.Tensor:
    """Load KGE entity embeddings — auto-detects dim from filename.

    Looks for hetionet_{model}_ent_*d.npy in kge_hetionet/results/baseline/{model}_hetionet/.
    """
    model = kge_type or KGE_TYPE
    result_dir = os.path.join(KGE_DIR, 'results', 'baseline', f'{model}_hetionet')
    if not os.path.isdir(result_dir):
        raise FileNotFoundError(
            f"KGE result dir not found: {result_dir}\n"
            f"Run kge_hetionet/run_all_kge.sh first.")
    npy_files = sorted(f for f in os.listdir(result_dir)
                       if f.endswith('.npy') and 'ent_' in f)
    if not npy_files:
        raise FileNotFoundError(
            f"No entity .npy found in {result_dir}\n"
            f"Run kge_hetionet/src/extract_embeddings.py first.")
    path = os.path.join(result_dir, npy_files[-1])
    print(f"  [KGE] Loading {path}")
    return torch.from_numpy(np.load(path)).float()


# Legacy alias
load_transe_embeddings = load_kge_embeddings


def build_model(cell: str, device: str) -> tuple:
    """Build DDIModel for the given cell ID.

    Returns (model, exclude_drugs, rel_is_pairwise).
    """
    hidden_dim = HP['hidden_dim']
    dropout = HP['dropout']
    num_classes = 86

    encoder = None
    rel_is_pairwise = False
    exclude_drugs = set()
    subgraph_extractor = None
    gat_cache = None
    drug2kg_table = None

    if cell == '01':
        # Cell 01: GAT on mol graph (55-dim)
        smiles_dict = load_drug_smiles()
        mol_graphs = load_mol_graphs_55d(smiles_dict)
        encoder = GATEncoder(hidden_dim=hidden_dim, dropout=dropout)
        encoder.set_features(mol_graphs)
        gat_cache = encoder._data_cache

    elif cell == '02':
        # Cell 02: SchNet on 3D coords (all 1710, drug 1050 = MANUAL_OCTAHEDRAL)
        coords_dict = load_schnet_coords()
        encoder = SchNetEncoder(hidden_dim=hidden_dim, dropout=dropout)
        encoder.set_features(coords_dict)

    elif cell == '03':
        # Cell 03: Graph Transformer (TIGER-style, 67-dim)
        smiles_dict = load_drug_smiles()
        mol_graphs = load_mol_graphs_tiger(smiles_dict)
        encoder = GraphTransformerMolEncoder(hidden_dim=hidden_dim, dropout=dropout)
        encoder.set_features(mol_graphs)

    elif cell == '04':
        # Cell 04: MLP on Morgan FP
        smiles_dict = load_drug_smiles()
        fps = load_drug_fingerprints(smiles_dict)
        encoder = MorganFPEncoder(hidden_dim=hidden_dim, dropout=dropout)
        encoder.set_features(fps)

    elif cell == '05':
        # Cell 05: FP Jaccard -> train-only PCA (DDIMDL, leakage-fixed)
        emb_dict = load_pca_embeddings('fp')
        encoder = BioBERTFrozenEncoder(hidden_dim=hidden_dim, dropout=dropout, bert_dim=128)
        encoder.set_features(emb_dict)

    elif cell == '06':
        # Cell 06: ChemBERTa frozen [CLS]
        emb_dict = load_chemberta_embeddings()
        encoder = ChemBERTaEncoder(hidden_dim=hidden_dim, dropout=dropout, bert_dim=384)
        encoder.set_features(emb_dict)

    elif cell == '07':
        # Cell 07: 1D CNN on SMILES
        smiles_dict = load_drug_smiles()
        encoder = SMILESCNNEncoder(hidden_dim=hidden_dim, dropout=dropout)
        encoder.set_features(smiles_dict)

    elif cell == '08':
        # Cell 08: KGE frozen lookup (kge_type selects model, auto-detects dim)
        kge_emb = load_kge_embeddings()
        kge_dim = kge_emb.shape[1]
        encoder = TransEEncoder(hidden_dim=hidden_dim, dropout=dropout,
                                num_entities=NUM_HETIONET_ENTITIES, embed_dim=kge_dim)
        with torch.no_grad():
            encoder.entity_embedding.weight.copy_(kge_emb)
        encoder.set_features(num_drugs=1710)  # identity mapping

    elif cell == '09':
        # Cell 09: R-GCN on full HetioNet subgraph + KGE frozen init
        # Full HetioNet = KG.txt (non-DDI, rel 86-108) + train.txt (DDI, rel 0-85)
        kge_emb = load_kge_embeddings()
        triples = load_hetionet_triples()   # 109 relation types, no offset
        encoder = RGCNSubgraphEncoder(
            hidden_dim=hidden_dim, dropout=dropout,
            num_entities=NUM_HETIONET_ENTITIES,
            num_relations=NUM_HETIONET_RELATIONS,   # 109
            num_bases=4, num_layers=2, edge_dropout=0.4,
            use_transe_init=True, input_dim=kge_emb.shape[1],
        )
        encoder.load_transe_init(kge_emb)
        encoder.set_features(num_drugs=1710)
        cache_dir = os.path.join(EXPERIMENT_DIR, 'data', 'subgraph_cache_full')
        encoder.set_kg_data(triples, NUM_HETIONET_ENTITIES, NUM_HETIONET_RELATIONS,
                            cache_dir=cache_dir)
        rel_is_pairwise = True
        subgraph_extractor = encoder.subgraph_extractor
        drug2kg_table = encoder._drug2kg_table

    elif cell == '10':
        # Cell 10: Graph Transformer on full HetioNet subgraph + KGE frozen init
        # Full HetioNet = KG.txt (non-DDI, rel 86-108) + train.txt (DDI, rel 0-85)
        kge_emb = load_kge_embeddings()
        triples = load_hetionet_triples()   # 109 relation types, no offset
        encoder = GraphTransformerKGEncoder(
            hidden_dim=hidden_dim, dropout=dropout,
            num_entities=NUM_HETIONET_ENTITIES,
            num_relations=NUM_HETIONET_RELATIONS,   # 109
            num_layers=2, num_heads=4,
            max_distance=8, max_degree=50,
            use_transe_init=True, input_dim=kge_emb.shape[1],
        )
        encoder.set_transe_init(kge_emb)
        encoder.set_features(num_drugs=1710)
        cache_dir = os.path.join(EXPERIMENT_DIR, 'data', 'subgraph_cache_full')
        encoder.set_kg_data(triples, NUM_HETIONET_ENTITIES, NUM_HETIONET_RELATIONS,
                            hop=2, max_nodes_per_hop=200, cache_dir=cache_dir)
        rel_is_pairwise = True
        subgraph_extractor = encoder.subgraph_extractor
        drug2kg_table = encoder._drug2kg_table

    elif cell == '12':
        # Cell 12: Interaction profile (bio) Jaccard -> train-only PCA (DDIMDL, leakage-fixed)
        # Profile: drug-target / enzyme / pathway from drugbank_attrs.
        # PCA output dim=384 -> Linear(384->128) = 49K trainable.
        # 112/1700 zero-profile drugs -> train PCA mean embedding (honest, no leakage)
        emb_dict = load_pca_embeddings('bio')
        encoder = BioBERTFrozenEncoder(hidden_dim=hidden_dim, dropout=dropout, bert_dim=384)
        encoder.set_features(emb_dict)
        # rel_is_pairwise stays False, exclude_drugs stays empty (REL cell pattern)

    elif cell == '11':
        # Cell 11: R-GCN on whole DDI graph (Decagon-style) + KGE fine-tunable init
        # Node table: 1710 (all drug IDs 0-1709). KGE rows 0-1709 = drug embeddings.
        # num_relations = 86 × 2 = 172 (forward + inverse). Fine-tunable.
        # First RGCNConv projects input_dim (KGE native dim) -> hidden_dim=128.
        kge_emb = load_kge_embeddings()
        drug_kge = kge_emb[:1710]  # identity: drug_id == entity_id
        edge_index, edge_type = build_ddi_edge_index(
            os.path.join(KG_DATA_DIR, 'train.txt'), num_drugs=1710)
        encoder = RGCNWholeDDIEncoder(
            hidden_dim=hidden_dim,
            num_drugs=1710,
            num_relations=172,
            num_bases=30,
            num_layers=2,
            dropout=dropout,
            edge_dropout=0.4,
            use_transe_init=True, input_dim=drug_kge.shape[1],
        )
        encoder.set_transe_init(drug_kge)
        encoder.set_ddi_graph(edge_index, edge_type)
        # rel_is_pairwise stays False (whole-graph lookup, no subgraph)
        # exclude_drugs stays empty (REL cell pattern, same as 08/09/10)

    elif cell == '13':
        # Cell 13: BioBERT frozen [CLS] (description) -> 768 -> 128
        emb_dict = load_biobert_embeddings('description')
        encoder = BioBERTFrozenEncoder(hidden_dim=hidden_dim, dropout=dropout, bert_dim=768)
        encoder.set_features(emb_dict)

    elif cell == '14':
        # Cell 14: BioBERT frozen [CLS] (drug name) -> 768 -> 128
        emb_dict = load_biobert_embeddings('drugname')
        encoder = BioBERTFrozenEncoder(hidden_dim=hidden_dim, dropout=dropout, bert_dim=768)
        encoder.set_features(emb_dict)

    else:
        raise ValueError(f"Unknown cell: {cell}. Valid: 01-14 (except 15-17 LLM cells).")

    # Determine source key (for DDIModel)
    if cell in ('08', '09', '10', '11', '12'):
        source_key = 'rel'
    elif cell in ('13', '14'):
        source_key = 'doc'
    else:
        source_key = 'mol'

    model = DDIModel(
        encoders={source_key: encoder},
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
        rel_is_pairwise=rel_is_pairwise,
    )

    return model, exclude_drugs, rel_is_pairwise, subgraph_extractor, gat_cache, drug2kg_table


def evaluate(model, loader, device, criterion):
    """Evaluate model on a DataLoader.

    Returns (metrics dict, y_true, y_pred, y_prob).
    """
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            loss = criterion(logits, batch.labels.long())
            total_loss += loss.item()
            num_batches += 1
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            all_labels.append(batch.labels.cpu().numpy())
            all_preds.append(preds)
            all_probs.append(probs)

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)

    y_true_unique = np.unique(y_true)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    acc      = accuracy_score(y_true, y_pred)
    kappa    = cohen_kappa_score(y_true, y_pred) if len(y_true_unique) > 1 else 0.0

    metrics = {
        'macro_f1': float(macro_f1),
        'acc':      float(acc),
        'kappa':    float(kappa),
        'loss':     total_loss / max(num_batches, 1),
    }

    return metrics, y_true, y_pred, y_prob


def save_results(
    result_dir, run_id,
    test_metrics, val_metrics,
    y_true, y_pred, y_prob,
    loss_history, config_dict,
    model, test_loader, device,
):
    """Save per-class F1 / predictions / loss_curves / pair_embeddings / config / summary."""

    # 1. config dump
    os.makedirs(os.path.join(result_dir, 'configs'), exist_ok=True)
    with open(os.path.join(result_dir, 'configs', f'{run_id}.yaml'), 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    # 2. per-class F1
    os.makedirs(os.path.join(result_dir, 'per_class'), exist_ok=True)
    pc_f1 = f1_score(y_true, y_pred, labels=list(range(86)), average=None, zero_division=0)
    with open(os.path.join(result_dir, 'per_class', f'{run_id}.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([f'class_{i}' for i in range(86)])
        w.writerow(pc_f1.tolist())

    # 3. predictions (y_prob)
    os.makedirs(os.path.join(result_dir, 'predictions'), exist_ok=True)
    np.save(os.path.join(result_dir, 'predictions', f'{run_id}.npy'), y_prob)

    # 4. loss_curves (per-epoch)
    if loss_history:
        os.makedirs(os.path.join(result_dir, 'loss_curves'), exist_ok=True)
        fieldnames = list(loss_history[0].keys())
        # run_id에서 split 제거 (loss curve는 전체 학습 공유)
        base_id = run_id.rsplit('_', 1)[0]
        lc_path = os.path.join(result_dir, 'loss_curves', f'{base_id}.csv')
        if not os.path.exists(lc_path):
            with open(lc_path, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(loss_history)

    # 5. pair embeddings
    model.eval()
    all_pairs = []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            try:
                pair = model.pair_embedding(batch)
                all_pairs.append(pair.cpu().numpy())
            except Exception:
                break
    if all_pairs:
        os.makedirs(os.path.join(result_dir, 'pair_embeddings'), exist_ok=True)
        np.save(os.path.join(result_dir, 'pair_embeddings', f'{run_id}.npy'),
                np.concatenate(all_pairs))


def train(
    cell: str,
    gpu: int = 0,
    seed: int = 0,
    epochs: int = None,
    result_dir: str = None,
    checkpoint_metric: str = 'macro_f1',
    smoke_pairs: int = None,
    skip_eval: bool = False,
):
    """Train one cell, evaluate on S0/S1/S2 simultaneously.

    Args:
        cell: cell ID string ('01'-'14')
        gpu: GPU index
        seed: random seed
        epochs: override HP['epochs'] if provided
        result_dir: where to save results (defaults to experiment/results/)
        checkpoint_metric: 'macro_f1' (default) or 'acc' — val metric for best-model selection
    """
    if epochs is not None:
        HP['epochs'] = epochs

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'
    if result_dir is None:
        result_dir = os.path.join(EXPERIMENT_DIR, 'results_drugbank')
    os.makedirs(result_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Cell {cell} | device={device} | seed={seed} | epochs={HP['epochs']}")
    print(f"{'='*60}")

    # Build model
    model, exclude_drugs, rel_is_pairwise, subgraph_extractor, gat_cache, drug2kg_table = \
        build_model(cell, device)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model params (trainable): {n_params:,}")

    # Build collate fn
    # Cell 10 (GT KG subgraph) needs _to_tiger_inputs applied in DataLoader workers
    # so the batch has sp_edge_* ready for forward_pair.
    subgraph_transform = None
    if cell == '10':
        subgraph_transform = model.encoders['rel']._to_tiger_inputs
    collate_fn = build_collate_fn(
        gat_cache=gat_cache,
        subgraph_extractor=subgraph_extractor,
        drug2kg_table=drug2kg_table,
        subgraph_transform=subgraph_transform,
    )

    # Cell 10: precompute all subgraphs (+tiger_inputs) before DataLoader creation.
    # Workers inherit populated cache via CoW fork → DataLoader = 0.004ms/pair.
    # Disk cache: first run saves to tiger_cache.pkl; subsequent runs load in <10s.
    # Skipped in smoke mode (smoke_pairs set) — 512 pairs run fine at 6ms each.
    if cell == '10' and subgraph_extractor is not None and smoke_pairs is None:
        import random as _random, pickle as _pickle
        _cache_dir = os.path.join(EXPERIMENT_DIR, 'data', 'subgraph_cache_full')
        _tiger_pkl = os.path.join(_cache_dir, 'tiger_subgraphs.pkl')

        if os.path.exists(_tiger_pkl):
            print(f"  [Cell 10] Loading tiger cache from disk: {_tiger_pkl}")
            with open(_tiger_pkl, 'rb') as _f:
                subgraph_extractor._cache = _pickle.load(_f)
            print(f"  [Cell 10] Loaded {len(subgraph_extractor._cache):,} cached subgraphs")
        else:
            print("  [Cell 10] Reading all pairs for subgraph precompute...")
            _all_pairs = []
            for _split in ['train', 'valid_S0', 'valid_S1', 'valid_S2',
                           'test_S0', 'test_S1', 'test_S2']:
                _fpath = os.path.join(DATA_DIR, f'{_split}.txt')
                if not os.path.exists(_fpath):
                    continue
                with open(_fpath) as _f:
                    for _line in _f:
                        _parts = _line.strip().split()
                        if len(_parts) == 3:
                            _h, _t = int(_parts[0]), int(_parts[1])
                            if _h not in exclude_drugs and _t not in exclude_drugs:
                                _all_pairs.append((_h, _t))
            print(f"  [Cell 10] Total pairs: {len(_all_pairs):,}")
            # Precompute subgraphs + _tiger_inputs (6.6ms/pair, ~21min one-time)
            subgraph_extractor.precompute_all(_all_pairs, transform_fn=subgraph_transform)
            # Save to disk for subsequent runs
            os.makedirs(_cache_dir, exist_ok=True)
            with open(_tiger_pkl, 'wb') as _f:
                _pickle.dump(subgraph_extractor._cache, _f)
            print(f"  [Cell 10] Tiger cache saved: {_tiger_pkl}")

        # Sanity check: sample 5 cached subgraphs
        _sample_keys = list(subgraph_extractor._cache.keys())[:5]
        _ok = True
        for _key in _sample_keys:
            _d = subgraph_extractor._cache[_key]
            if _d is None or _d.num_nodes < 2 or _d.edge_index.shape[1] == 0:
                print(f"  [Cell 10] WARN: degenerate subgraph {_key}")
                _ok = False
        if _ok:
            _tiger_ok = all(hasattr(subgraph_extractor._cache[k], '_tiger_inputs')
                            for k in _sample_keys)
            print(f"  [Cell 10] Sanity check PASSED — "
                  f"nodes={[subgraph_extractor._cache[k].num_nodes for k in _sample_keys]}, "
                  f"tiger_precomputed={_tiger_ok}")

    # Build dataloaders
    train_loader, valid_loaders, test_loaders = build_dataloaders(
        data_dir=DATA_DIR,
        split_types=['S0', 'S1', 'S2'],
        batch_size=HP['batch_size'],
        num_workers=4,
        exclude_drugs=exclude_drugs,
        collate_fn=collate_fn,
    )
    # Smoke-mode: subset training data to limit DataLoader CPU time
    if smoke_pairs is not None and smoke_pairs < len(train_loader.dataset):
        from torch.utils.data import Subset, DataLoader as TorchDataLoader
        subset_ds = Subset(train_loader.dataset, list(range(smoke_pairs)))
        train_loader = TorchDataLoader(
            subset_ds,
            batch_size=HP['batch_size'],
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn,
        )
    print(f"  Train: {len(train_loader.dataset)} pairs")
    for s in ['S0', 'S1', 'S2']:
        print(f"  Valid_{s}: {len(valid_loaders[s].dataset)} | "
              f"Test_{s}: {len(test_loaders[s].dataset)}")

    # Loss, optimizer (DDI-Bench 원본: no scheduler)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=HP['lr'], weight_decay=HP['weight_decay'],
    )

    assert checkpoint_metric in ('macro_f1', 'acc'), \
        f"checkpoint_metric must be 'macro_f1' or 'acc', got '{checkpoint_metric}'"

    # Per-split best tracking (full 200 epochs, no early stopping)
    splits = ['S0', 'S1', 'S2']
    best_val_f1  = {s: -1.0 for s in splits}
    best_epoch   = {s: 0    for s in splits}
    best_states  = {s: None for s in splits}
    loss_history = []

    start_time = time.time()

    for epoch in range(1, HP['epochs'] + 1):
        # Train
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, batch.labels.long())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        train_loss = total_loss / max(n_batches, 1)

        # Validate all splits, track per-split best (full 200 epochs)
        epoch_row = {'epoch': epoch, 'train_loss': train_loss}
        if not skip_eval:
            for s in splits:
                val_metrics, _, _, _ = evaluate(model, valid_loaders[s], device, criterion)
                score = val_metrics[checkpoint_metric]
                epoch_row[f'val_loss_{s}']     = val_metrics['loss']
                epoch_row[f'val_macro_f1_{s}'] = val_metrics['macro_f1']
                epoch_row[f'val_acc_{s}']      = val_metrics['acc']
                if score > best_val_f1[s]:
                    best_val_f1[s] = score
                    best_epoch[s]  = epoch
                    best_states[s] = {
                        k: v.detach().cpu().clone()
                        for k, v in model.state_dict().items()
                    }
        loss_history.append(epoch_row)

        if epoch % 10 == 0 or epoch == 1:
            bests_str = ' | '.join(
                f"{s}={best_val_f1[s]:.4f}@{best_epoch[s]}" for s in splits)
            print(f"  Ep {epoch:3d} | loss={train_loss:.4f} | val best: {bests_str}")

    wall_time = time.time() - start_time
    print(f"\n  Training done in {wall_time:.0f}s")

    # Final test evaluation: per-split best checkpoint
    if skip_eval:
        # Smoke mode: skip final evaluation entirely
        print("  [skip_eval] Final evaluation skipped.")
        return

    config_dict = {
        'cell': cell, 'seed': seed, 'n_params': n_params,
        'checkpoint_metric': checkpoint_metric,
        **{f'hp_{k}': v for k, v in HP.items()},
    }
    results = {}
    for s in splits:
        if best_states[s] is not None:
            model.load_state_dict(best_states[s])
            model.to(device)
        val_metrics,  _, _, _             = evaluate(model, valid_loaders[s], device, criterion)
        test_metrics, y_true, y_pred, y_prob = evaluate(model, test_loaders[s],  device, criterion)
        results[s] = {
            'val': val_metrics, 'test': test_metrics, 'best_epoch': best_epoch[s],
        }
        print(f"  [{s}] best_ep={best_epoch[s]} | "
              f"val_f1={val_metrics['macro_f1']:.4f} | "
              f"test_f1={test_metrics['macro_f1']:.4f}")

        cm_tag = checkpoint_metric[:3]  # 'mac' or 'acc'
        run_id = f"cell{cell}_seed{seed}_cm{cm_tag}_{s}"
        save_results(
            result_dir=result_dir,
            run_id=run_id,
            test_metrics=test_metrics,
            val_metrics=val_metrics,
            y_true=y_true, y_pred=y_pred, y_prob=y_prob,
            loss_history=loss_history,
            config_dict={**config_dict, 'split': s, 'best_epoch': best_epoch[s]},
            model=model,
            test_loader=test_loaders[s],
            device=device,
        )

    # Summary CSV
    summary_path = os.path.join(result_dir, 'summary.csv')
    rows = []
    for s in splits:
        row = {
            'cell': cell, 'split': s, 'seed': seed,
            'checkpoint_metric': checkpoint_metric,
            'best_epoch': results[s]['best_epoch'],
            'n_params': n_params,
            'wall_time_sec': round(wall_time, 1),
        }
        for k, v in results[s]['test'].items():
            row[f'test_{k}'] = round(v, 6) if isinstance(v, float) else v
        for k, v in results[s]['val'].items():
            row[f'val_{k}'] = round(v, 6) if isinstance(v, float) else v
        rows.append(row)

    file_exists = os.path.exists(summary_path)
    with open(summary_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

    # ntfy 완료 알림
    best_s0 = results['S0']['test']['macro_f1']
    best_s1 = results['S1']['test']['macro_f1']
    best_s2 = results['S2']['test']['macro_f1']
    ntfy(f"[V5] Cell {cell} seed{seed} done | "
         f"S0={best_s0:.4f} S1={best_s1:.4f} S2={best_s2:.4f} | {wall_time:.0f}s")

    print(f"  Results saved to: {summary_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description='V5 DDI Cell Trainer')
    parser.add_argument('--cell', type=str, required=True,
                        help='Cell ID: 01-14 (15-17 = LLM cells, separate script)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU index (default: 0)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed (default: 0)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override epochs (default: 200)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size (default: 128)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate (default: 3e-4)')
    parser.add_argument('--result_dir', type=str, default=None,
                        help='Result directory (default: experiment/results_drugbank/)')
    parser.add_argument('--checkpoint_metric', type=str, default='macro_f1',
                        choices=['macro_f1', 'acc'],
                        help='Val metric for best-model selection (default: macro_f1)')
    parser.add_argument('--smoke_pairs', type=int, default=None,
                        help='Limit training to first N pairs (smoke test only)')
    parser.add_argument('--skip_eval', action='store_true', default=False,
                        help='Skip validation during training (smoke test only)')
    args = parser.parse_args()

    # Apply overrides
    if args.batch_size is not None:
        HP['batch_size'] = args.batch_size
    if args.lr is not None:
        HP['lr'] = args.lr

    # Pad cell to 2 digits
    cell = args.cell.zfill(2)

    train(
        cell=cell,
        gpu=args.gpu,
        seed=args.seed,
        epochs=args.epochs,
        result_dir=args.result_dir,
        checkpoint_metric=args.checkpoint_metric,
        smoke_pairs=args.smoke_pairs,
        skip_eval=args.skip_eval,
    )


if __name__ == '__main__':
    main()
