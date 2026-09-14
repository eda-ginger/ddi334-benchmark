"""
DDI-334 TWOSIDES Multi-Label Trainer — DDI-Bench 원본 프로토콜 그대로
=====================================================================
loss / 평가(predict) / checkpoint(update_result) / loss_weight 는 DDI-Bench
원본(`original_repo/.../DDI_Ben/`)의 코드를 **그대로** 옮겼다. 우리 review
encoder(GAT/SchNet/GT/MLP/CNN/ChemBERTa/BioBERT)만 모델로 꽂는다.

원본 대응:
  loss          : models/MLP|Decagon/model.py  loss()  (twosides 분기)
                  BCELoss(weight=loss_weight)(sigmoid(pred)*vec, vec*polarity)
  loss_weight   : trainer.py:44  occur.min()/occur   (train positive triplet 빈도)
  평가          : trainer.py:200-216 (twosides)  type별 ROC-AUC/PR-AUC/accuracy 평균
  checkpoint    : trainer.py:220-231 update_result  val accuracy 기준, split별 best,
                  세 split 전부 patience 초과 시 종료
  HP            : main.py argparse  AdamW/lr3e-4/wd1e-5/batch128/patience10, scheduler 없음

데이터: ddi334/{ddibn,tdc}/  ("h t y0,...,y(N-1) p", db_id 공간, pos/neg 1:1 교차)
Usage:
  micromamba run -n DDIBench python train.py --cell 04 --dataset ddibn --gpu 0 --seed 42
"""
import os
import sys
import argparse
import json
import csv
import time
import subprocess
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, f1_score

# ── 경로 (스크립트는 ddi334/ 안) ──
DDI334_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPERIMENT_DIR = os.path.dirname(DDI334_DIR)
sys.path.insert(0, EXPERIMENT_DIR)              # models/, dataio/ 임포트용
PRECOMPUTE_DIR = os.path.join(EXPERIMENT_DIR, 'data', 'precompute')
KGE_DIR = os.path.join(DDI334_DIR, 'data', 'kge')       # REL KGE 임베딩 (데이터셋별)
NUM_HETIONET_ENTITIES = 34124

from models.encoders import (
    GATEncoder, MorganFPEncoder, SchNetEncoder,             # MOL 구조 (01/04/02)
    GraphTransformerMolEncoder, SMILESCNNEncoder,           # MOL 구조 (03/07)
    FPJaccardEncoder, ChemBERTaEncoder,                     # frozen (05/06)
    InteractionJaccardEncoder, BioBERTEncoder,              # frozen (12 / 13·14)
    TransEEncoder, RGCNWholeDDIEncoder,                     # REL (08/11)
    RGCNSubgraphEncoder, GraphTransformerKGEncoder,         # REL (09/10)
    BioProfileMLPEncoder,                                   # REL raw MLP (15)
)
from models.ddi_model import DDIModel
from dataio.dataset import DDIBatch


def load_kge_embeddings(dataset, kge_type='transe'):
    """KGE entity 임베딩 [34124, dim]. {ds}/precompute/{type}_ent.npy 우선, 없으면 kge/{ds}/{type}/.
    CASE3 env=1이면 kge/{ds}_case3/{type}/ 사용(신약 고립 KG로 학습한 임베딩, precompute 무시)."""
    if os.environ.get('DDISPLIT'):   # 진단: DDI-split 학습 임베딩 (Ideal=ddibn_ddisplit / Real=+CASE3=ddibn_ddisplit_case3)
        sub = f'{dataset}_ddisplit_case3' if os.environ.get('CASE3') else f'{dataset}_ddisplit'
        d = os.path.join(KGE_DIR, sub, kge_type)
        npys = sorted(f for f in os.listdir(d) if f.endswith('.npy') and 'ent_' in f)
        if not npys:
            raise FileNotFoundError(f"DDISPLIT KGE .npy 없음: {d}")
        print(f"  [KGE/DDISPLIT{'/CASE3' if os.environ.get('CASE3') else ''}] {os.path.join(d, npys[-1])}")
        return torch.from_numpy(np.load(os.path.join(d, npys[-1]))).float()
    if os.environ.get('CASE3'):
        d = os.path.join(KGE_DIR, f'{dataset}_case3', kge_type)
        npys = sorted(f for f in os.listdir(d) if f.endswith('.npy') and 'ent_' in f)
        if not npys:
            raise FileNotFoundError(f"CASE3 KGE .npy 없음: {d}")
        print(f"  [KGE/CASE3] {os.path.join(d, npys[-1])}")
        return torch.from_numpy(np.load(os.path.join(d, npys[-1]))).float()
    if os.environ.get('ABLNR'):   # ablation: Case-1 임베딩 그대로 + novel 행만 random (재학습 X)
        d = os.path.join(KGE_DIR, f'{dataset}_ablnr', kge_type)
        npys = sorted(f for f in os.listdir(d) if f.endswith('.npy') and 'ent_' in f)
        if not npys:
            raise FileNotFoundError(f"ABLNR KGE .npy 없음: {d}")
        print(f"  [KGE/ABLNR] {os.path.join(d, npys[-1])}")
        return torch.from_numpy(np.load(os.path.join(d, npys[-1]))).float()
    pc = os.path.join(PRECOMPUTE_DIR, f'{kge_type}_ent.npy')
    if os.path.exists(pc):
        print(f"  [KGE] {pc}")
        return torch.from_numpy(np.load(pc)).float()
    d = os.path.join(KGE_DIR, dataset, kge_type)
    npys = sorted(f for f in os.listdir(d) if f.endswith('.npy') and 'ent_' in f)
    if not npys:
        raise FileNotFoundError(f"KGE .npy 없음: {pc} / {d}")
    print(f"  [KGE] {os.path.join(d, npys[-1])}")
    return torch.from_numpy(np.load(os.path.join(d, npys[-1]))).float()

NTFY_TOPIC = 'Latex_project'


def ntfy(msg: str):
    try:
        subprocess.Popen(['curl', '-s', '-d', msg, f'ntfy.sh/{NTFY_TOPIC}'],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


# ── HP (DDI-Bench 원본) ──
HP = {
    'hidden_dim': 128,
    'lr': 3e-4,             # main.py default 0.0003
    'weight_decay': 1e-5,   # main.py default
    'epochs': 200,          # 상한 (실제 종료는 patience)
    'batch_size': 128,      # main.py default
    'dropout': 0.2,
    'patience': 10,         # main.py default
}

NUM_TYPES = None  # 데이터에서 결정


# ── Dataset: "h t vec p" -> (d1, d2, label[N+1]=vec|p) ──
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
                labels.append(vec + [p])     # [N+1] (DDI-Bench true_label 구조)
        self.pairs = np.array(pairs, dtype=np.int64)
        self.labels = np.array(labels, dtype=np.float32)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        return int(self.pairs[i, 0]), int(self.pairs[i, 1]), self.labels[i]


def build_collate(gat_cache=None, subgraph_extractor=None,
                  drug2kg_table=None, subgraph_transform=None):
    """gat_cache: Cell 01 mol graph. subgraph_*: Cell 09/10 KG subgraph (v5 build_collate_fn)."""
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


# ── Feature loaders (train.py와 동일, db_id 키) ──
def load_drug_smiles():
    with open(os.path.join(PRECOMPUTE_DIR, 'drug_smiles.json')) as f:
        return {int(k): v for k, v in json.load(f).items()}


def load_drug_fingerprints(smiles_dict):
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


def load_mol_graphs_55d(smiles_dict):
    from models.encoders import mol_to_graph_55d
    return {did: mol_to_graph_55d(smi) for did, smi in smiles_dict.items()}


def load_mol_graphs_tiger(smiles_dict):
    from models.encoders import mol_to_graph_tiger
    return {did: mol_to_graph_tiger(smi) for did, smi in smiles_dict.items()}


def load_schnet_coords():
    return torch.load(os.path.join(PRECOMPUTE_DIR, 'schnet_coords.pt'), map_location='cpu', weights_only=False)


def load_chemberta_embeddings():
    return torch.load(os.path.join(PRECOMPUTE_DIR, 'chemberta.pt'), map_location='cpu', weights_only=False)['embeddings']


def load_biobert_embeddings(input_type='description'):
    # real/genes/ideal (2026-07-27, 05_실험설계(DDI334).md §8): LLM(§5) 템플릿과 통일한 cell13 3-tier.
    #   build_biobert_template.py로 생성. DOC_MODEL env(biobert/pubmedbert/scibert)로 backbone 선택
    #   (2026-07-27 확장, §8.5). description/drugname/smiles_c3/rdkitdesc_c3는 구설계(legacy, biobert 고정).
    # 반환: (embeddings dict, hidden_dim) — 모델마다 실제 dim을 payload에서 읽어옴(하드코딩 금지).
    legacy = {'description': 'biobert_description.pt', 'drugname': 'biobert_drugname.pt',
              'smiles_c3': 'biobert_smiles.pt', 'rdkitdesc_c3': 'biobert_rdkitdesc.pt'}
    if input_type in legacy:
        fname = legacy[input_type]
    else:
        doc_model = os.environ.get('DOC_MODEL', 'biobert')
        fname = f'{doc_model}_{input_type}.pt'
    payload = torch.load(os.path.join(PRECOMPUTE_DIR, fname), map_location='cpu', weights_only=False)
    return payload['embeddings'], payload['hidden_dim']


def load_pca_embeddings(kind):
    """Cell 05 fp / Cell 12 bio PCA. PRECOMPUTE_DIR({ds}/precompute). returns (emb_dict, input_dim)."""
    fname = 'fp_pca.pt' if kind == 'fp' else 'bio_pca.pt'
    pl = torch.load(os.path.join(PRECOMPUTE_DIR, fname), map_location='cpu', weights_only=False)
    return pl['embeddings'], pl['input_dim']


def load_raw_bio_profile():
    """Cell 15: raw HetioNet bio-neighbor multi-hot vector (build_bio_raw.py). No similarity/PCA."""
    pl = torch.load(os.path.join(PRECOMPUTE_DIR, 'bio_raw.pt'), map_location='cpu', weights_only=False)
    return pl['embeddings'], pl['input_dim']


# ── Model builder (cells 01-07, 13-14) — train_ddi334.py와 동일 ──
def build_model(cell, device, drug_ids, dataset=None):
    hd, dr = HP['hidden_dim'], HP['dropout']
    encoder, gat_cache, exclude = None, None, set()
    rel_ctx = None   # Cell 09/10 subgraph: {extractor, drug2kg_table, transform}

    if cell == '01':
        # GAT는 GT(Cell 03)와 동일한 TIGER 67-d 노드 피처 사용 (노드 피처 통일, 아키텍처만 변수)
        g = load_mol_graphs_tiger(load_drug_smiles())
        encoder = GATEncoder(hidden_dim=hd, dropout=dr, atom_feat_dim=67); encoder.set_features(g); gat_cache = encoder._data_cache
    elif cell == '02':
        coords = load_schnet_coords()
        encoder = SchNetEncoder(hidden_dim=hd, dropout=dr); encoder.set_features(coords)
        for db_id in drug_ids:
            if db_id not in coords:   # 334 전부 3D 좌표 있음(검증) -> exclude 0. schnet_drop 불필요
                exclude.add(db_id)
    elif cell == '03':
        g = load_mol_graphs_tiger(load_drug_smiles())
        encoder = GraphTransformerMolEncoder(hidden_dim=hd, dropout=dr); encoder.set_features(g)
    elif cell == '04':
        fps = load_drug_fingerprints(load_drug_smiles())
        encoder = MorganFPEncoder(hidden_dim=hd, dropout=dr); encoder.set_features(fps)
    elif cell == '05':
        emb, dim = load_pca_embeddings('fp')
        encoder = FPJaccardEncoder(hidden_dim=hd, dropout=dr, bert_dim=dim); encoder.set_features(emb)
    elif cell == '06':
        encoder = ChemBERTaEncoder(hidden_dim=hd, dropout=dr, bert_dim=384); encoder.set_features(load_chemberta_embeddings())
    elif cell == '07':
        encoder = SMILESCNNEncoder(hidden_dim=hd, dropout=dr); encoder.set_features(load_drug_smiles())
    elif cell == '13':
        # DOC_INPUT env로 입력 텍스트 선택. 2026-07-27~: real/genes/ideal(LLM 템플릿과 통일, §8) 사용.
        # description/smiles_c3/rdkitdesc_c3는 구설계(legacy, 재현용으로만 남김).
        # DOC_MODEL env(biobert/pubmedbert/scibert, §8.5)로 backbone 선택 — bert_dim은 payload에서 동적으로 읽음.
        doc_inp = os.environ.get('DOC_INPUT', 'description')
        emb, bert_dim = load_biobert_embeddings(doc_inp)
        encoder = BioBERTEncoder(hidden_dim=hd, dropout=dr, bert_dim=bert_dim); encoder.set_features(emb)
    elif cell == '14':
        # legacy(drug name 전용, Real 카운터파트 없음) — cell13 real/genes/ideal에 흡수됨. 재현용으로만 유지.
        emb, bert_dim = load_biobert_embeddings('drugname')
        encoder = BioBERTEncoder(hidden_dim=hd, dropout=dr, bert_dim=bert_dim); encoder.set_features(emb)
    # ── REL 셀 ──
    elif cell == '08':
        # KGE frozen lookup (TransE 기본; KGE_TYPE 환경변수로 5종 선택). db_id==entity_id (identity).
        kge_type = os.environ.get('KGE_TYPE', 'transe')
        kge_emb = load_kge_embeddings(dataset, kge_type)
        encoder = TransEEncoder(hidden_dim=hd, dropout=dr,
                                num_entities=NUM_HETIONET_ENTITIES, embed_dim=kge_emb.shape[1])
        with torch.no_grad():
            encoder.entity_embedding.weight.copy_(kge_emb)
        encoder.set_features(num_drugs=1710)
    elif cell == '11':
        # R-GCN on whole DDI graph (Decagon) + TransE fine-tunable init. subgraph 불필요.
        # node=db_id (identity, num_drugs=1710 cover). rel=type_idx+inverse (ddi_whole.pt).
        kge_type = os.environ.get('KGE_TYPE', 'transe')
        kge_emb = load_kge_embeddings(dataset, kge_type)
        drug_kge = kge_emb[:1710]   # rows 0-1709 = drug slots (db_id==entity_id)
        whole = torch.load(os.path.join(PRECOMPUTE_DIR, 'ddi_whole.pt'), weights_only=False)
        encoder = RGCNWholeDDIEncoder(
            hidden_dim=hd, num_drugs=1710, num_relations=int(whole['num_relations']),
            num_bases=30, num_layers=2, dropout=dr, edge_dropout=0.4,
            use_transe_init=True, input_dim=drug_kge.shape[1])
        encoder.set_transe_init(drug_kge)
        encoder.set_ddi_graph(whole['edge_index'], whole['edge_type'])
    elif cell in ('09', '10'):
        # KG subgraph (bio 0-22 + TWOSIDES train 23+). node=db_id (identity, drug2kg=None).
        # num_relations = n_relations(non-doubled, 232/1331); 추출기가 inverse로 내부 doubling.
        # 캐시: {ds}/precompute/subgraph_cache/ (데이터셋별 KG가 달라 반드시 분리)
        kge_type = os.environ.get('KGE_TYPE', 'transe')
        kge_emb = load_kge_embeddings(dataset, kge_type)
        num_ent = kge_emb.shape[0]
        rm = json.load(open(os.path.join(KGE_DIR, dataset, 'relation_map.json')))
        num_rel = int(rm['n_relations'])                       # 232 / 1331
        if os.environ.get('CASE3'):   # 신약 고립 KG: novel edge 제거된 triples + 별도 캐시(case1 재사용 금지)
            triples = np.load(os.path.join(DDI334_DIR, 'data', 'case3', 'kg_triples_case3.npy'))   # [E,3] (h,t,r)
            cache_dir = os.path.join(PRECOMPUTE_DIR, 'subgraph_cache_case3')
        else:
            triples = np.load(os.path.join(PRECOMPUTE_DIR, 'kg_triples.npy'))   # [E,3] (h,t,r)
            cache_dir = os.path.join(PRECOMPUTE_DIR, 'subgraph_cache')
        if cell == '09':
            encoder = RGCNSubgraphEncoder(
                hidden_dim=hd, dropout=dr, num_entities=num_ent, num_relations=num_rel,
                num_bases=4, num_layers=2, edge_dropout=0.4,
                use_transe_init=True, input_dim=kge_emb.shape[1])
            encoder.load_transe_init(kge_emb)
            encoder.set_features(num_drugs=1710)
            encoder.set_kg_data(triples, num_ent, num_rel, cache_dir=cache_dir)
            transform = None
        else:  # '10'
            encoder = GraphTransformerKGEncoder(
                hidden_dim=hd, dropout=dr, num_entities=num_ent, num_relations=num_rel,
                num_layers=2, num_heads=4, max_distance=8, max_degree=50,
                use_transe_init=True, input_dim=kge_emb.shape[1])
            encoder.set_transe_init(kge_emb)
            encoder.set_features(num_drugs=1710)
            encoder.set_kg_data(triples, num_ent, num_rel,
                                hop=2, max_nodes_per_hop=200, cache_dir=cache_dir)
            transform = encoder._to_tiger_inputs
        rel_ctx = {'extractor': encoder.subgraph_extractor,
                   'drug2kg_table': encoder._drug2kg_table,
                   'transform': transform}
    elif cell == '12':
        # Bio interaction profile -> DDI-334 train-only PCA (fit=Dk 270). frozen feature.
        emb, dim = load_pca_embeddings('bio')
        encoder = InteractionJaccardEncoder(hidden_dim=hd, dropout=dr, bert_dim=dim)
        encoder.set_features(emb)
    elif cell == '15':
        # Bio interaction profile, raw multi-hot (no Jaccard/PCA) -> 2-layer MLP,
        # mirrors Cell 04 (MorganFPEncoder). build_bio_raw.py precompute.
        emb, dim = load_raw_bio_profile()
        encoder = BioProfileMLPEncoder(hidden_dim=hd, dropout=dr, fp_dim=dim)
        encoder.set_features(emb)
    else:
        raise ValueError(f"Cell '{cell}' not supported (01-15)")

    if cell in ('08', '09', '10', '11', '12', '15'):
        source_key = 'rel'
    elif cell in ('13', '14'):
        source_key = 'doc'
    else:
        source_key = 'mol'
    model = DDIModel(encoders={source_key: encoder}, hidden_dim=hd,
                     num_classes=NUM_TYPES, dropout=dr,
                     rel_is_pairwise=(rel_ctx is not None))
    return model, exclude, gat_cache, rel_ctx


# ════════════════════════════════════════════════════════════════════════════
# DDI-Bench 원본 그대로: loss / 평가 / checkpoint
# ════════════════════════════════════════════════════════════════════════════
def make_loss(loss_weight, device):
    """models/MLP|Decagon/model.py loss() (twosides) 그대로."""
    bceloss = nn.BCELoss(weight=torch.tensor(loss_weight, dtype=torch.float32, device=device))

    def loss_fn(pred, true_label):
        vec = true_label[:, :-1]
        pol = true_label[:, -1].unsqueeze(1)
        return bceloss(torch.sigmoid(pred) * vec, vec * pol)
    return loss_fn


def compute_loss_weight(train_ds):
    """trainer.py:44  occur = train positive vec 합;  loss_weight = occur.min()/occur."""
    lab = train_ds.labels                      # [M, N+1]
    pol = lab[:, -1:]                          # positive flag
    occur = (lab[:, :-1] * pol).sum(0)         # train positive triplet 빈도 [N]
    pos = occur[occur > 0]
    mn = pos.min() if pos.size > 0 else 1.0
    lw = np.zeros_like(occur)
    lw[occur > 0] = mn / occur[occur > 0]      # occur=0 type은 0 (train 미등장)
    return lw


@torch.no_grad()
def evaluate(model, loader, device, want_per_type=False, want_raw=False):
    """trainer.py:200-216 (twosides) 그대로: type별 ROC-AUC/PR-AUC/accuracy 평균.

    추가 지표 (2026-06-17): type별 ranking precision 두 가지를 함께 산출.
      - P@50    : 점수 내림차순 상위 50개(가용<50이면 전부) 중 실제 양성 비율 (Decagon/KnowDDI식 고정 컷).
                  TWOSIDES 평가는 type별 1:1 균형 -> 후보가 50 미만인 type은 base-rate(~0.5)로 수렴.
      - P@(N/2) : 상위 (후보수//2)개 중 양성 비율 = R-Precision (EmerGNN base_model.py:161 k=len//2).
                  type 크기에 적응하므로 균형 데이터에서 원리적.
    want_per_type=True면 (metrics, per_type[N] dict)도 반환 (per_class 저장용).
    want_raw=True면 (..., pred_final[M,N], label_final[M,N+1])도 반환 (raw 점수 .npz 저장용 = 향후 재학습 불필요).
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
        lc = label_final[where, j] * label_final[where, -1]   # = polarity (해당 행)
        if where.shape[0] > 0 and 0 < lc.sum() < lc.shape[0]:
            phard = (pc > 0.5).astype('float')
            r = roc_auc_score(lc, pc); p = average_precision_score(lc, pc)
            a = accuracy_score(lc, phard)
            fv = f1_score(lc, phard, zero_division=0)   # type별 binary F1 (양성=실제 상호작용)
            roc.append(r); prc.append(p); ap.append(a); f1.append(fv)
            roc_pt[j] = r; pr_pt[j] = p; ap_pt[j] = a; f1_pt[j] = fv
            # ranking precision: 점수 내림차순 정렬 후 상위 k개 양성 비율
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


# ── Training ──
def train(cell, dataset, gpu=0, seed=0, epochs=None, result_dir=None, apk=False):
    global NUM_TYPES
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu'
    if epochs:
        HP['epochs'] = epochs

    data_dir = os.path.join(DDI334_DIR, 'data', dataset)
    global PRECOMPUTE_DIR
    PRECOMPUTE_DIR = os.path.join(data_dir, 'precompute')   # 모든 MOL/REL/DOC feature를 여기서 로드
    meta = json.load(open(os.path.join(data_dir, 'meta.json')))
    NUM_TYPES = meta['n_types']
    drug_ids = [int(v) for v in json.load(open(os.path.join(data_dir, 'drug_map.json'))).values()]

    if result_dir is None:
        result_dir = os.path.join(DDI334_DIR, 'results', dataset)
    os.makedirs(result_dir, exist_ok=True)

    print(f"\n[DDI-334/{dataset}] Cell {cell} | N={NUM_TYPES} | {device} | seed={seed}")
    ntfy(f"[DDI334/{dataset}] Cell{cell} seed{seed} START | N={NUM_TYPES}")

    model, exclude, gat_cache, rel_ctx = build_model(cell, device, drug_ids, dataset=dataset)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if exclude:
        print(f"  excluded {len(exclude)} drugs (no 3D coords)")

    # datasets 먼저 (subgraph precompute에 전 split 쌍 필요)
    train_ds = DDI334Dataset(data_dir, 'train', exclude)
    splits = ['S0', 'S1', 'S2']
    vds, tds = {}, {}
    for s in splits:
        vds[s] = DDI334Dataset(data_dir, f'valid_{s}', exclude)
        tds[s] = DDI334Dataset(data_dir, f'test_{s}', exclude)

    # Cell 09/10: 학습 전 전체 약물쌍 subgraph 미리 추출 -> 캐시 (worker가 CoW fork로 공유)
    if rel_ctx is not None:
        tbl = rel_ctx['drug2kg_table']
        uniq = set()
        for ds_ in [train_ds] + list(vds.values()) + list(tds.values()):
            for a, b in ds_.pairs:
                uniq.add((int(tbl[int(a)]), int(tbl[int(b)])))
        print(f"  [subgraph] precompute {len(uniq)} distinct pairs -> {os.path.join(PRECOMPUTE_DIR, 'subgraph_cache')}")
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

    # checkpoint: split별 best (val ROC-AUC 기준 — 사용자 확정), trainer.py update_result 골격
    CKPT_METRIC = 'accuracy'   # 2026-06-17: primary=accuracy 전환 (인코더+LLM 공정비교 동일 지표 + 선택=보고 일관). 구 'AUC-ROC' 결과는 results/ddibn_ckpt-rocauc 보관
    best_val = {s: -1.0 for s in splits}
    best_epoch = {s: 0 for s in splits}
    best_state = {s: None for s in splits}
    no_improve = {s: 0 for s in splits}
    loss_history = []          # v2식 loss_curves
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
    tag = f"cell{cell}_{dataset}_seed{seed}"
    # 2026-06-17: 전 지표(ROC/PR/Acc/F1/P@50/P@(N/2)) + raw 점수(.npz) 항상 저장 (apk 인자는 하위호환 무시)
    for sub in ['loss_curves', 'per_class', 'configs', 'raw_preds']:
        os.makedirs(os.path.join(result_dir, sub), exist_ok=True)

    # loss_curves
    with open(os.path.join(result_dir, 'loss_curves', f'{tag}.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(loss_history[0].keys()))
        w.writeheader(); w.writerows(loss_history)

    # 최종 test: split별 best 로드 후 평가 (+ per_class, config, summary, raw_preds)
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
            f.write(f"cell: '{cell}'\ndataset: {dataset}\nsplit: {s}\nseed: {seed}\n"
                    f"checkpoint_metric: {CKPT_METRIC}\nbest_epoch: {best_epoch[s]}\n"
                    f"n_params: {n_params}\nn_types: {NUM_TYPES}\n"
                    f"hp_lr: {HP['lr']}\nhp_weight_decay: {HP['weight_decay']}\n"
                    f"hp_batch_size: {HP['batch_size']}\nhp_dropout: {HP['dropout']}\n"
                    f"hp_hidden_dim: {HP['hidden_dim']}\nhp_epochs: {HP['epochs']}\n"
                    f"hp_patience: {HP['patience']}\nhp_optimizer: AdamW\n")
        row = {'cell': cell, 'dataset': dataset, 'seed': seed, 'split': s,
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
    ntfy(f"[DDI334/{dataset}] Cell{cell} seed{seed} DONE {wall:.0f}s\n" +
         " ".join(f"{r['split']}:acc{r['test_accuracy']:.3f}/f1{r['test_macro-F1']:.3f}" for r in rows))
    print(f"  saved -> {summary} (+loss_curves/per_class/configs)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', required=True)
    ap.add_argument('--dataset', required=True, choices=['ddibn', 'tdc'])
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--result_dir', default=None)
    ap.add_argument('--apk', action='store_true',
                    help='재현 검증 + P@50/P@(N/2) 산출 + raw 점수 .npz 저장 (summary_apk.csv 별도 기록)')
    args = ap.parse_args()
    train(args.cell, args.dataset, args.gpu, args.seed, args.epochs, args.result_dir, apk=args.apk)


if __name__ == '__main__':
    main()
