"""
DDI-334 TWOSIDES Multi-Label Trainer -- follows the original DDI-Bench protocol as-is
=====================================================================
loss / evaluation (predict) / checkpoint (update_result) / loss_weight are carried
over **as-is** from the original DDI-Bench code (`original_repo/.../DDI_Ben/`). Only
our review encoders (GAT/SchNet/GT/MLP/CNN/ChemBERTa/BioBERT) are plugged in as the
model.

Correspondence to the original:
  loss          : models/MLP|Decagon/model.py  loss()  (twosides branch)
                  BCELoss(weight=loss_weight)(sigmoid(pred)*vec, vec*polarity)
  loss_weight   : trainer.py:44  occur.min()/occur   (train positive-triplet frequency)
  evaluation    : trainer.py:200-216 (twosides)  per-type ROC-AUC/PR-AUC/accuracy, averaged
  checkpoint    : trainer.py:220-231 update_result  best per split by val accuracy;
                  stops once all three splits exceed patience
  HP            : main.py argparse  AdamW/lr3e-4/wd1e-5/batch128/patience10, no scheduler

Data: ddi334/data/splits/  ("h t y0,...,y(N-1) p", db_id space, pos/neg 1:1 crossed)
          data/molecule, data/relation, data/document -- per-modality precomputed inputs
Usage:
  micromamba run -n DDIBench python train/train.py --cell 04 --gpu 0 --seed 42
"""
import os
import sys
import argparse
import json
import csv
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, f1_score

# -- Paths (this script lives in ddi334/code/; encoders.py/ddi_model.py/dataset.py are in the same folder) --
DDI334_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_TAG = 'ddibn'   # Constant kept only for result filename / summary.csv schema compatibility.
                        # Not used to build data/ paths. (There's only one dataset so the --dataset
                        # flag was removed, but the value is kept to stay format-compatible with
                        # already-committed results/*/summary.csv and loss_curves filenames.)
SPLITS_DIR = os.path.join(DDI334_DIR, 'data', 'splits')        # train/valid/test splits + metadata
MOL_DIR = os.path.join(DDI334_DIR, 'data', 'molecule')         # Molecule-modality inputs
REL_DIR = os.path.join(DDI334_DIR, 'data', 'relation')         # Relation-modality inputs
DOC_DIR = os.path.join(DDI334_DIR, 'data', 'document')         # Document-modality inputs
KGE_DIR = os.path.join(REL_DIR, 'kge')                         # Relation KGE embeddings (existing_drug/novel_drug)
NUM_HETIONET_ENTITIES = 34124

from encoders import (
    GATEncoder, MorganFPEncoder, SchNetEncoder,             # Molecule structure (04/03/06)
    GraphTransformerMolEncoder, SMILESCNNEncoder,           # Molecule structure (05/01)
    ChemBERTaEncoder,                                        # frozen (02)
    BioBERTEncoder,                                          # frozen (12)
    TransEEncoder, RGCNWholeDDIEncoder,                     # Relation (09/08)
    RGCNSubgraphEncoder, GraphTransformerKGEncoder,         # Relation (10/11)
    BioProfileMLPEncoder,                                   # Relation raw MLP (07)
)
from ddi_model import DDIModel
from dataset import DDIBatch


def load_kge_embeddings(kge_type='transe'):
    """KGE entity embeddings [34124, dim]. Default: kge/existing_drug/{type}/.
    If env NOVEL_DRUG=1, use kge/novel_drug/{type}/ (embeddings trained on the
    novel-drug-isolated KG). If env EMB_SRC ('existing_drug'/'novel_drug') is set,
    it forces the embedding source regardless of NOVEL_DRUG (for the Existing/Novel
    subgraph x Existing/Novel TransE cross ablation, added 2026-10-19)."""
    emb_src = os.environ.get('EMB_SRC')
    if emb_src in ('novel_drug', 'existing_drug'):
        d = os.path.join(KGE_DIR, emb_src, kge_type)
        npys = sorted(f for f in os.listdir(d) if f.endswith('.npy') and 'ent_' in f)
        if not npys:
            raise FileNotFoundError(f"EMB_SRC={emb_src}: no KGE .npy found: {d}")
        print(f"  [KGE/EMB_SRC={emb_src}] {os.path.join(d, npys[-1])}")
        return torch.from_numpy(np.load(os.path.join(d, npys[-1]))).float()
    if os.environ.get('NOVEL_DRUG'):
        d = os.path.join(KGE_DIR, 'novel_drug', kge_type)
        npys = sorted(f for f in os.listdir(d) if f.endswith('.npy') and 'ent_' in f)
        if not npys:
            raise FileNotFoundError(f"NOVEL_DRUG: no KGE .npy found: {d}")
        print(f"  [KGE/NOVEL_DRUG] {os.path.join(d, npys[-1])}")
        return torch.from_numpy(np.load(os.path.join(d, npys[-1]))).float()
    if os.environ.get('ABLNR'):   # ablation: keep existing_drug embeddings as-is, randomize only the novel-drug rows (no retraining)
        d = os.path.join(KGE_DIR, 'ablnr', kge_type)
        npys = sorted(f for f in os.listdir(d) if f.endswith('.npy') and 'ent_' in f)
        if not npys:
            raise FileNotFoundError(f"ABLNR: no KGE .npy found: {d}")
        print(f"  [KGE/ABLNR] {os.path.join(d, npys[-1])}")
        return torch.from_numpy(np.load(os.path.join(d, npys[-1]))).float()
    d = os.path.join(KGE_DIR, 'existing_drug', kge_type)
    npys = sorted(f for f in os.listdir(d) if f.endswith('.npy') and 'ent_' in f)
    if not npys:
        raise FileNotFoundError(f"No KGE .npy found: {d}")
    print(f"  [KGE] {os.path.join(d, npys[-1])}")
    return torch.from_numpy(np.load(os.path.join(d, npys[-1]))).float()

# -- HP (original DDI-Bench values) --
HP = {
    'hidden_dim': 128,
    'lr': 3e-4,             # main.py default 0.0003
    'weight_decay': 1e-5,   # main.py default
    'epochs': 200,          # upper bound (actual stop is via patience)
    'batch_size': 128,      # main.py default
    'dropout': 0.2,
    'patience': 10,         # main.py default
}

NUM_TYPES = None  # determined from the data


# -- Dataset: "h t vec p" -> (d1, d2, label[N+1]=vec|p) --
class DDI334Dataset(Dataset):
    def __init__(self, data_dir, split, exclude=None):
        path = os.path.join(data_dir, f'{split}.txt')
        exclude = exclude or set()
        pairs, labels = [], []
        with open(path) as f:
            for line in f:
                parts = line.split()
                if len(parts) != 4:
                    continue
                d1, d2 = int(parts[0]), int(parts[1])
                if d1 in exclude or d2 in exclude:
                    continue
                vec = [int(x) for x in parts[2].split(',')]
                p = int(parts[3])
                pairs.append((d1, d2))
                labels.append(vec + [p])     # [N+1] (DDI-Bench true_label structure)
        self.pairs = np.array(pairs, dtype=np.int64)
        self.labels = np.array(labels, dtype=np.float32)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        return int(self.pairs[i, 0]), int(self.pairs[i, 1]), self.labels[i]


def build_collate(gat_cache=None, subgraph_extractor=None,
                  drug2kg_table=None, subgraph_transform=None):
    """gat_cache: Cell 04 mol graph. subgraph_*: Cell 10/11 KG subgraph (v5 build_collate_fn)."""
    from torch_geometric.data import Batch as PyGBatch

    def collate(samples):
        d1 = [s[0] for s in samples]
        d2 = [s[1] for s in samples]
        lab = torch.stack([torch.from_numpy(s[2]) for s in samples])  # [B, N+1]
        d1_t = torch.tensor(d1, dtype=torch.long)
        d2_t = torch.tensor(d2, dtype=torch.long)
        mb1 = mb2 = None
        if gat_cache is not None:
            mb1 = PyGBatch.from_data_list([gat_cache[d] for d in d1])
            mb2 = PyGBatch.from_data_list([gat_cache[d] for d in d2])
        rel_sg = None
        if subgraph_extractor is not None:
            if drug2kg_table is not None:
                kg1 = drug2kg_table[d1_t].tolist()
                kg2 = drug2kg_table[d2_t].tolist()
            else:
                kg1, kg2 = d1, d2
            dl = [subgraph_extractor.extract(k1, k2) for k1, k2 in zip(kg1, kg2)]
            if subgraph_transform is not None:
                dl = [subgraph_transform(d) for d in dl]
            rel_sg = PyGBatch.from_data_list(dl)
        return DDIBatch(drug1_ids=d1_t, drug2_ids=d2_t, labels=lab,
                        mol_batch_d1=mb1, mol_batch_d2=mb2, rel_subgraph_batch=rel_sg)
    return collate


# -- Feature loaders (same as train.py, keyed by db_id) --
def load_drug_smiles():
    with open(os.path.join(MOL_DIR, 'drug_smiles.json')) as f:
        return {int(k): v for k, v in json.load(f).items()}


def load_drug_fingerprints(smiles_dict):
    """Used by load_molfp_embeddings() below (NODE_INIT=molfp) only — Cell 03's own
    Fingerprint input is precomputed by build/molecule/fingerprint.py instead, see
    load_fingerprint_embeddings()."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    fps = {}
    for did, smi in smiles_dict.items():
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fps[did] = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024), dtype=np.float32)
        else:
            fps[did] = np.zeros(1024, dtype=np.float32)
    return fps


def load_molfp_embeddings():
    """Initial node feature based on Morgan FP (1024d, radius2) (NODE_INIT=molfp, added 2026-10-19).
    Uses the db_id==entity_id identity: only drug rows (0-1709) are filled with molecular
    structure; non-drug HetioNet entities (genes/diseases etc., 1710+) are left at 0 --
    an experiment contrasting with the R-GCN subgraph (cell10) TransE initialization, under
    the premise that "a drug's own molecular information is always available regardless of
    the Existing/Novel condition"."""
    cache = os.path.join(MOL_DIR, 'molfp_ent_1024d.npy')
    if os.path.exists(cache):
        print(f"  [NODE_INIT/molfp] {cache}")
        return torch.from_numpy(np.load(cache)).float()
    smiles = load_drug_smiles()
    fps = load_drug_fingerprints(smiles)
    emb = np.zeros((NUM_HETIONET_ENTITIES, 1024), dtype=np.float32)
    for did, fp in fps.items():
        if did < NUM_HETIONET_ENTITIES:
            emb[did] = fp
    np.save(cache, emb)
    print(f"  [NODE_INIT/molfp] computed -> {cache} ({len(fps)} drugs)")
    return torch.from_numpy(emb).float()


def load_mol_graphs_2d():
    """Cell 04/05: TIGER-style 2D graph (build/molecule/mol_graph.py precompute)."""
    return torch.load(os.path.join(MOL_DIR, 'mol_graph_2d.pt'), map_location='cpu', weights_only=False)


def load_schnet_coords():
    return torch.load(os.path.join(MOL_DIR, 'schnet_coords.pt'), map_location='cpu', weights_only=False)


def load_chemberta_embeddings():
    return torch.load(os.path.join(MOL_DIR, 'chemberta.pt'), map_location='cpu', weights_only=False)['embeddings']


def load_fingerprint_embeddings():
    """Cell 03: Morgan fingerprint (build/molecule/fingerprint.py precompute)."""
    emb = torch.load(os.path.join(MOL_DIR, 'fingerprint.pt'), map_location='cpu', weights_only=False)['embeddings']
    return {k: v.numpy() for k, v in emb.items()}


def load_biobert_embeddings(input_type='existing_drug'):
    # novel_drug/existing_drug (2026-07-27, 05_실험설계(DDI334).md section 8): cell12
    #   input unified with the LLM (section 5) templates. Generated by mention_description.py.
    #   Backbone selected via DOC_MODEL env (biobert/pubmedbert/scibert, default scibert = the
    #   manuscript's Table 1/D.1 setting) (extended 2026-07-27, section 8.5; biobert/pubmedbert
    #   are for the Appendix C.1 comparison).
    # Returns: (embeddings dict, hidden_dim) -- each model's actual dim is read from the
    #   payload (never hardcoded).
    doc_model = os.environ.get('DOC_MODEL', 'scibert')
    fname = f'{doc_model}_{input_type}.pt'
    payload = torch.load(os.path.join(DOC_DIR, fname), map_location='cpu', weights_only=False)
    return payload['embeddings'], payload['hidden_dim']


def load_raw_bio_profile():
    """Cell 07: raw HetioNet bio-neighbor multi-hot vector (bio_profile.py). No similarity/PCA."""
    pl = torch.load(os.path.join(REL_DIR, 'bio_profile.pt'), map_location='cpu', weights_only=False)
    return pl['embeddings'], pl['input_dim']


# -- Model builder -- uses the manuscript's Table 2 numbering (01-12) as-is (2026-10-21: internal cell numbers rearranged to match the manuscript) --
def build_model(cell, device, drug_ids):
    hd, dr = HP['hidden_dim'], HP['dropout']
    encoder, gat_cache, exclude = None, None, set()
    rel_ctx = None   # Cell 10/11 subgraph: {extractor, drug2kg_table, transform}

    # -- Molecule cells (01-06) --
    if cell == '01':
        # SMILES (1D-CNN)
        encoder = SMILESCNNEncoder(hidden_dim=hd, dropout=dr); encoder.set_features(load_drug_smiles())
    elif cell == '02':
        # SMILES (ChemBERTa, frozen)
        encoder = ChemBERTaEncoder(hidden_dim=hd, dropout=dr, bert_dim=384); encoder.set_features(load_chemberta_embeddings())
    elif cell == '03':
        # Fingerprint (MLP)
        fps = load_fingerprint_embeddings()
        encoder = MorganFPEncoder(hidden_dim=hd, dropout=dr); encoder.set_features(fps)
    elif cell == '04':
        # Molecular graph (GAT). Uses the same TIGER 67-d node features as the Graph Transformer (Cell 05) -- node features unified, only the architecture varies
        g = load_mol_graphs_2d()
        encoder = GATEncoder(hidden_dim=hd, dropout=dr, atom_feat_dim=67); encoder.set_features(g); gat_cache = encoder._data_cache
    elif cell == '05':
        # Molecular graph (Graph Transformer)
        g = load_mol_graphs_2d()
        encoder = GraphTransformerMolEncoder(hidden_dim=hd, dropout=dr); encoder.set_features(g)
    elif cell == '06':
        # Molecular graph (SchNet)
        coords = load_schnet_coords()
        encoder = SchNetEncoder(hidden_dim=hd, dropout=dr); encoder.set_features(coords)
        for db_id in drug_ids:
            if db_id not in coords:   # all 334 have 3D coords (verified) -> exclude stays empty. schnet_drop not needed
                exclude.add(db_id)
    # -- Relation cells (07-11) --
    elif cell == '07':
        # Bio profile (MLP). raw multi-hot (no Jaccard/PCA) -> 2-layer MLP,
        # mirrors Cell 03 (MorganFPEncoder). bio_profile.py precompute.
        emb, dim = load_raw_bio_profile()
        encoder = BioProfileMLPEncoder(hidden_dim=hd, dropout=dr, fp_dim=dim)
        encoder.set_features(emb)
    elif cell == '08':
        # DDI network (R-GCN), whole graph + frozen init. No subgraph needed.
        # node=db_id (identity, covers num_drugs=1710). rel=type_idx+inverse (ddi_network.pt).
        # NODE_INIT env ('molfp' default / 'transe') selects the initial node-value source.
        # This graph contains only drug nodes (no genes/diseases) -- unlike the cell10/11 KG subgraph.
        # 2026-10-19: since molfp (Morgan FP, 4.66M params) was adopted as the manuscript's official
        # result for "DDI network (R-GCN)", the default switched from transe->molfp (cell10/11 keep
        # transe as their default). To reproduce the old TransE results, set NODE_INIT=transe explicitly.
        node_init = os.environ.get('NODE_INIT', 'molfp')
        if node_init == 'molfp':
            drug_kge = load_molfp_embeddings()[:1710]
        else:
            kge_type = os.environ.get('KGE_TYPE', 'transe')
            kge_emb = load_kge_embeddings(kge_type)
            drug_kge = kge_emb[:1710]   # rows 0-1709 = drug slots (db_id==entity_id)
        whole = torch.load(os.path.join(REL_DIR, 'ddi_network.pt'), weights_only=False)
        encoder = RGCNWholeDDIEncoder(
            hidden_dim=hd, num_drugs=1710, num_relations=int(whole['num_relations']),
            num_bases=30, num_layers=2, dropout=dr, edge_dropout=0.4,
            use_transe_init=True, input_dim=drug_kge.shape[1])
        encoder.set_transe_init(drug_kge)
        encoder.set_ddi_graph(whole['edge_index'], whole['edge_type'])
    elif cell == '09':
        # Knowledge graph (KGE, frozen lookup). TransE is the default; select via the KGE_TYPE env var. db_id==entity_id (identity).
        kge_type = os.environ.get('KGE_TYPE', 'transe')
        kge_emb = load_kge_embeddings(kge_type)
        encoder = TransEEncoder(hidden_dim=hd, dropout=dr,
                                num_entities=NUM_HETIONET_ENTITIES, embed_dim=kge_emb.shape[1])
        with torch.no_grad():
            encoder.entity_embedding.weight.copy_(kge_emb)
        encoder.set_features(num_drugs=1710)
    elif cell in ('10', '11'):
        # Knowledge graph (subgraph): 10=R-GCN / 11=Graph Transformer.
        # KG subgraph (bio 0-22 + TWOSIDES train 23+). node=db_id (identity, drug2kg=None).
        # num_relations = n_relations (non-doubled, 232/1331); the extractor doubles it internally via inverse relations.
        # Cache: data/relation/subgraph_cache/
        # NODE_INIT env ('transe' default / 'molfp') selects the initial node-feature source. SUBGRAPH_SRC env
        # ('existing_drug'/'novel_drug') forces only the subgraph (graph connectivity) regardless of NOVEL_DRUG --
        # for the Existing/Novel embedding x Existing/Novel subgraph cross ablation plus the Morgan FP
        # initialization experiment (added 2026-10-19; if neither is set, behavior is unchanged).
        node_init = os.environ.get('NODE_INIT', 'transe')
        if node_init == 'molfp':
            init_emb = load_molfp_embeddings()
        else:
            kge_type = os.environ.get('KGE_TYPE', 'transe')
            init_emb = load_kge_embeddings(kge_type)
        num_ent = init_emb.shape[0]
        rm = json.load(open(os.path.join(KGE_DIR, 'existing_drug', 'relation_map.json')))
        num_rel = int(rm['n_relations'])                       # 232 / 1331
        subgraph_src = os.environ.get('SUBGRAPH_SRC')
        if subgraph_src == 'existing_drug':
            use_novel_drug_subgraph = False
        elif subgraph_src == 'novel_drug':
            use_novel_drug_subgraph = True
        else:
            use_novel_drug_subgraph = bool(os.environ.get('NOVEL_DRUG'))
        if use_novel_drug_subgraph:   # Novel-drug-isolated KG: triples with novel edges removed + a separate cache (do not reuse the existing_drug one)
            triples = np.load(os.path.join(KGE_DIR, 'kg_triples_novel_drug.npy'))   # [E,3] (h,t,r)
            cache_dir = os.path.join(REL_DIR, 'subgraph_cache_novel_drug')
        else:
            triples = np.load(os.path.join(REL_DIR, 'kg_triples.npy'))   # [E,3] (h,t,r)
            cache_dir = os.path.join(REL_DIR, 'subgraph_cache')
        if cell == '10':
            encoder = RGCNSubgraphEncoder(
                hidden_dim=hd, dropout=dr, num_entities=num_ent, num_relations=num_rel,
                num_bases=4, num_layers=2, edge_dropout=0.4,
                use_transe_init=True, input_dim=init_emb.shape[1])
            encoder.load_transe_init(init_emb)
            encoder.set_features(num_drugs=1710)
            encoder.set_kg_data(triples, num_ent, num_rel, cache_dir=cache_dir)
            transform = None
        else:  # '11'
            encoder = GraphTransformerKGEncoder(
                hidden_dim=hd, dropout=dr, num_entities=num_ent, num_relations=num_rel,
                num_layers=2, num_heads=4, max_distance=8, max_degree=50,
                use_transe_init=True, input_dim=init_emb.shape[1])
            encoder.set_transe_init(init_emb)
            encoder.set_features(num_drugs=1710)
            encoder.set_kg_data(triples, num_ent, num_rel,
                                hop=2, max_nodes_per_hop=200, cache_dir=cache_dir)
            transform = encoder._to_tiger_inputs
        rel_ctx = {'extractor': encoder.subgraph_extractor,
                   'drug2kg_table': encoder._drug2kg_table,
                   'transform': transform}
    # -- Document cell (12) --
    elif cell == '12':
        # Description/Mention (SciBERT, frozen). DOC_INPUT env selects the input text.
        # One of novel_drug/existing_drug (unified with the LLM templates, section 8), default
        # existing_drug = the manuscript's Table 2/D.1 setting. DOC_MODEL env (biobert/pubmedbert/
        # scibert, section 8.5) selects the backbone, default scibert = the manuscript setting
        # (biobert/pubmedbert are for the Appendix C.1 comparison) -- bert_dim is read dynamically
        # from the payload.
        doc_inp = os.environ.get('DOC_INPUT', 'existing_drug')
        emb, bert_dim = load_biobert_embeddings(doc_inp)
        encoder = BioBERTEncoder(hidden_dim=hd, dropout=dr, bert_dim=bert_dim); encoder.set_features(emb)
    else:
        raise ValueError(f"Cell '{cell}' not supported (01-12)")

    if cell in ('07', '08', '09', '10', '11'):
        source_key = 'rel'
    elif cell == '12':
        source_key = 'doc'
    else:
        source_key = 'mol'
    model = DDIModel(encoders={source_key: encoder}, hidden_dim=hd,
                     num_classes=NUM_TYPES, dropout=dr,
                     rel_is_pairwise=(rel_ctx is not None))
    return model, exclude, gat_cache, rel_ctx


# ================================================================================
# Carried over as-is from the original DDI-Bench: loss / evaluation / checkpoint
# ================================================================================
def make_loss(loss_weight, device):
    """Exactly models/MLP|Decagon/model.py loss() (twosides)."""
    bceloss = nn.BCELoss(weight=torch.tensor(loss_weight, dtype=torch.float32, device=device))

    def loss_fn(pred, true_label):
        vec = true_label[:, :-1]
        pol = true_label[:, -1].unsqueeze(1)
        return bceloss(torch.sigmoid(pred) * vec, vec * pol)
    return loss_fn


def compute_loss_weight(train_ds):
    """trainer.py:44  occur = sum of train positive vecs;  loss_weight = occur.min()/occur."""
    lab = train_ds.labels                      # [M, N+1]
    pol = lab[:, -1:]                          # positive flag
    occur = (lab[:, :-1] * pol).sum(0)         # train positive-triplet frequency [N]
    pos = occur[occur > 0]
    mn = pos.min() if pos.size > 0 else 1.0
    lw = np.zeros_like(occur)
    lw[occur > 0] = mn / occur[occur > 0]      # types with occur=0 (absent from train) get 0
    return lw


@torch.no_grad()
def evaluate(model, loader, device, want_per_type=False, want_raw=False):
    """Exactly trainer.py:200-216 (twosides): per-type ROC-AUC/PR-AUC/accuracy, averaged.

    Additional metrics (2026-06-17): also computes two per-type ranking-precision variants.
      - P@50    : fraction of true positives among the top 50 by descending score (all of
                  them if fewer than 50 are available) -- the fixed-cutoff convention used by
                  Decagon/KnowDDI. TWOSIDES evaluation is 1:1 balanced per type, so types with
                  fewer than 50 candidates converge to the base rate (~0.5).
      - P@(N/2) : fraction of positives among the top (num_candidates // 2) = R-Precision
                  (EmerGNN base_model.py:161, k=len//2). Adapts to type size, so it's more
                  principled on balanced data.
    If want_per_type=True, also returns (metrics, per_type[N] dict) (for saving per_class).
    If want_raw=True, also returns (..., pred_final[M,N], label_final[M,N+1]) (for saving the
    raw-score .npz, so retraining isn't needed later).
    """
    model.eval()
    preds, labels = [], []
    for batch in loader:
        batch = batch.to(device)
        pred = torch.sigmoid(model(batch))     # [B, N]
        preds.append(pred.cpu().numpy())
        labels.append(batch.labels.cpu().numpy())   # [B, N+1]
    pred_final = np.concatenate(preds)         # [M, N]
    label_final = np.concatenate(labels)       # [M, N+1]
    N = pred_final.shape[1]
    roc_pt = np.zeros(N); pr_pt = np.zeros(N); ap_pt = np.zeros(N)
    f1_pt = np.zeros(N); p50_pt = np.zeros(N); prn_pt = np.zeros(N)
    roc, prc, ap, f1, p50, prn = [], [], [], [], [], []
    for j in range(N):
        where = np.where(label_final[:, j] == 1)[0]
        pc = pred_final[where, j]
        lc = label_final[where, j] * label_final[where, -1]   # = polarity (for this row)
        if where.shape[0] > 0 and 0 < lc.sum() < lc.shape[0]:
            phard = (pc > 0.5).astype('float')
            r = roc_auc_score(lc, pc); p = average_precision_score(lc, pc)
            a = accuracy_score(lc, phard)
            fv = f1_score(lc, phard, zero_division=0)   # per-type binary F1 (positive = actual interaction)
            roc.append(r); prc.append(p); ap.append(a); f1.append(fv)
            roc_pt[j] = r; pr_pt[j] = p; ap_pt[j] = a; f1_pt[j] = fv
            # ranking precision: sort by descending score, then fraction of positives among the top k
            lc_sorted = lc[np.argsort(-pc)]
            k50 = min(50, lc_sorted.shape[0])
            krn = max(1, lc_sorted.shape[0] // 2)
            v50 = float(lc_sorted[:k50].mean()); vrn = float(lc_sorted[:krn].mean())
            p50.append(v50); prn.append(vrn)
            p50_pt[j] = v50; prn_pt[j] = vrn
    metrics = {'AUC-ROC': float(np.mean(roc)) if roc else 0.0,
               'PR-AUC': float(np.mean(prc)) if prc else 0.0,
               'accuracy': float(np.mean(ap)) if ap else 0.0,
               'macro-F1': float(np.mean(f1)) if f1 else 0.0,
               'P@50': float(np.mean(p50)) if p50 else 0.0,
               'P@N2': float(np.mean(prn)) if prn else 0.0}
    out = (metrics,)
    if want_per_type:
        out += ({'roc': roc_pt, 'pr': pr_pt, 'acc': ap_pt, 'f1': f1_pt,
                 'p50': p50_pt, 'prn': prn_pt},)
    if want_raw:
        out += (pred_final, label_final)
    return out if len(out) > 1 else metrics


# -- Training --
def train(cell, gpu=0, seed=0, epochs=None, result_dir=None, apk=False):
    global NUM_TYPES
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'
    if epochs:
        HP['epochs'] = epochs

    meta = json.load(open(os.path.join(SPLITS_DIR, 'meta.json')))
    NUM_TYPES = meta['n_types']
    drug_ids = [int(v) for v in json.load(open(os.path.join(SPLITS_DIR, 'drug_map.json'))).values()]

    if result_dir is None:
        result_dir = os.path.join(DDI334_DIR, 'results', DATASET_TAG)
    os.makedirs(result_dir, exist_ok=True)

    print(f"\n[DDI-334/{DATASET_TAG}] Cell {cell} | N={NUM_TYPES} | {device} | seed={seed}")

    model, exclude, gat_cache, rel_ctx = build_model(cell, device, drug_ids)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if exclude:
        print(f"  excluded {len(exclude)} drugs (no 3D coords)")

    # datasets first (subgraph precompute needs every split's pairs)
    train_ds = DDI334Dataset(SPLITS_DIR, 'train', exclude)
    splits = ['S0', 'S1', 'S2']
    vds, tds = {}, {}
    for s in splits:
        vds[s] = DDI334Dataset(SPLITS_DIR, f'valid_{s}', exclude)
        tds[s] = DDI334Dataset(SPLITS_DIR, f'test_{s}', exclude)

    # Cell 10/11: extract subgraphs for all drug pairs before training -> cache (shared across workers via CoW fork)
    if rel_ctx is not None:
        tbl = rel_ctx['drug2kg_table']
        uniq = set()
        for ds_ in [train_ds] + list(vds.values()) + list(tds.values()):
            for a, b in ds_.pairs:
                uniq.add((int(tbl[int(a)]), int(tbl[int(b)])))
        print(f"  [subgraph] precompute {len(uniq)} distinct pairs -> {os.path.join(REL_DIR, 'subgraph_cache')}")
        rel_ctx['extractor'].precompute_all(list(uniq), transform_fn=rel_ctx['transform'])

    collate = build_collate(
        gat_cache=gat_cache,
        subgraph_extractor=(rel_ctx['extractor'] if rel_ctx else None),
        drug2kg_table=(rel_ctx['drug2kg_table'] if rel_ctx else None),
        subgraph_transform=(rel_ctx['transform'] if rel_ctx else None))
    valid_ld, test_ld = {}, {}
    for s in splits:
        vd, td = vds[s], tds[s]
        valid_ld[s] = DataLoader(vd, batch_size=HP['batch_size'], shuffle=False, num_workers=4, collate_fn=collate, pin_memory=True)
        test_ld[s] = DataLoader(td, batch_size=HP['batch_size'], shuffle=False, num_workers=4, collate_fn=collate, pin_memory=True)
    train_ld = DataLoader(train_ds, batch_size=HP['batch_size'], shuffle=True, num_workers=4, collate_fn=collate, pin_memory=True)
    print(f"  train {len(train_ds)} | " + " | ".join(f"val_{s} {len(valid_ld[s].dataset)}/test {len(test_ld[s].dataset)}" for s in splits))

    loss_fn = make_loss(compute_loss_weight(train_ds), device)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                      lr=HP['lr'], weight_decay=HP['weight_decay'])

    # checkpoint: best per split (by val ROC-AUC -- confirmed by the user), mirrors trainer.py update_result
    CKPT_METRIC = 'accuracy'   # 2026-06-17: switched primary metric to accuracy (same metric for a fair encoder+LLM comparison, and selection matches what's reported). Old 'AUC-ROC' results are kept at results/ddibn_ckpt-rocauc
    best_val = {s: -1.0 for s in splits}
    best_epoch = {s: 0 for s in splits}
    best_state = {s: None for s in splits}
    no_improve = {s: 0 for s in splits}
    loss_history = []          # v2-style loss_curves
    start = time.time()

    for epoch in range(1, HP['epochs'] + 1):
        model.train()
        losses = []
        for batch in train_ld:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch), batch.labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        row = {'epoch': epoch, 'train_loss': float(np.mean(losses))}
        stop_flags, msg = [], []
        for s in splits:
            v = evaluate(model, valid_ld[s], device)
            row[f'val_AUC-ROC_{s}'] = v['AUC-ROC']
            row[f'val_PR-AUC_{s}'] = v['PR-AUC']
            row[f'val_acc_{s}'] = v['accuracy']
            if v[CKPT_METRIC] > best_val[s]:
                best_val[s] = v[CKPT_METRIC]; best_epoch[s] = epoch; no_improve[s] = 0
                best_state[s] = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
            else:
                no_improve[s] += 1
            stop_flags.append(no_improve[s] > HP['patience'])
            msg.append(f"{s} roc={v['AUC-ROC']:.4f}@best{best_val[s]:.4f}")
        loss_history.append(row)
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Ep{epoch:3d} loss={row['train_loss']:.4f} | " + " | ".join(msg))
        if all(stop_flags):
            print(f"  Early stop at epoch {epoch} (all splits > patience={HP['patience']})")
            break

    wall = time.time() - start
    tag = f"cell{cell}_{DATASET_TAG}_seed{seed}"
    # 2026-06-17: always save every metric (ROC/PR/Acc/F1/P@50/P@(N/2)) + raw scores (.npz); the apk arg is ignored, kept only for backward compatibility
    for sub in ['loss_curves', 'per_class', 'configs', 'raw_preds']:
        os.makedirs(os.path.join(result_dir, sub), exist_ok=True)

    # loss_curves
    with open(os.path.join(result_dir, 'loss_curves', f'{tag}.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(loss_history[0].keys()))
        w.writeheader(); w.writerows(loss_history)

    # Final test: load each split's best checkpoint and evaluate (+ per_class, config, summary, raw_preds)
    rows = []
    for s in splits:
        if best_state[s] is not None:
            model.load_state_dict(best_state[s]); model.to(device)
        vm = evaluate(model, valid_ld[s], device)
        tm, per_type, pred_f, label_f = evaluate(model, test_ld[s], device,
                                                 want_per_type=True, want_raw=True)
        np.savez_compressed(os.path.join(result_dir, 'raw_preds', f'{tag}_{s}.npz'),
                            pred=pred_f.astype('float32'), label=label_f.astype('int8'))
        print(f"  [{s}] best_ep={best_epoch[s]} | val acc={vm['accuracy']:.4f} f1={vm['macro-F1']:.4f} "
              f"roc={vm['AUC-ROC']:.4f} | test acc={tm['accuracy']:.4f} f1={tm['macro-F1']:.4f} "
              f"roc={tm['AUC-ROC']:.4f} pr={tm['PR-AUC']:.4f} p50={tm['P@50']:.4f} pN2={tm['P@N2']:.4f}")

        # per_class (test per-type ROC-AUC / PR-AUC / accuracy / F1 / P@50 / P@(N/2))
        with open(os.path.join(result_dir, 'per_class', f'{tag}_{s}.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow([f'type_{j}' for j in range(NUM_TYPES)])
            for key in ('roc', 'pr', 'acc', 'f1', 'p50', 'prn'):
                w.writerow([round(float(x), 6) for x in per_type[key]])
        # config yaml
        with open(os.path.join(result_dir, 'configs', f'{tag}_{s}.yaml'), 'w') as f:
            f.write(f"cell: '{cell}'\ndataset: {DATASET_TAG}\nsplit: {s}\nseed: {seed}\n"
                    f"checkpoint_metric: {CKPT_METRIC}\nbest_epoch: {best_epoch[s]}\n"
                    f"n_params: {n_params}\nn_types: {NUM_TYPES}\n"
                    f"hp_lr: {HP['lr']}\nhp_weight_decay: {HP['weight_decay']}\n"
                    f"hp_batch_size: {HP['batch_size']}\nhp_dropout: {HP['dropout']}\n"
                    f"hp_hidden_dim: {HP['hidden_dim']}\nhp_epochs: {HP['epochs']}\n"
                    f"hp_patience: {HP['patience']}\nhp_optimizer: AdamW\n")
        row = {'cell': cell, 'dataset': DATASET_TAG, 'seed': seed, 'split': s,
               'checkpoint_metric': CKPT_METRIC, 'best_epoch': best_epoch[s],
               'n_params': n_params, 'wall_time_sec': round(wall, 1),
               'test_accuracy': round(tm['accuracy'], 6), 'test_macro-F1': round(tm['macro-F1'], 6),
               'test_AUC-ROC': round(tm['AUC-ROC'], 6), 'test_PR-AUC': round(tm['PR-AUC'], 6),
               'test_P@50': round(tm['P@50'], 6), 'test_P@N2': round(tm['P@N2'], 6),
               'val_accuracy': round(vm['accuracy'], 6), 'val_macro-F1': round(vm['macro-F1'], 6),
               'val_AUC-ROC': round(vm['AUC-ROC'], 6), 'val_PR-AUC': round(vm['PR-AUC'], 6),
               'val_P@50': round(vm['P@50'], 6), 'val_P@N2': round(vm['P@N2'], 6)}
        rows.append(row)

    summary = os.path.join(result_dir, 'summary.csv')
    exists = os.path.exists(summary)
    with open(summary, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
        if not exists:
            w.writeheader()
        w.writerows(rows)
    print(f"  saved -> {summary} (+loss_curves/per_class/configs)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', required=True)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--result_dir', default=None)
    ap.add_argument('--apk', action='store_true',
                    help='reproduction check + compute P@50/P@(N/2) + save raw-score .npz (logged separately to summary_apk.csv)')
    args = ap.parse_args()
    train(args.cell, args.gpu, args.seed, args.epochs, args.result_dir, apk=args.apk)


if __name__ == '__main__':
    main()
