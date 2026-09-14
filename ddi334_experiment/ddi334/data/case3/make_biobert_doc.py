"""Case-3 DOC: BioBERT [CLS] 임베딩 2변형 (구조 유도 텍스트).

원본 data/precompute/scripts/biobert.py와 동일 방식 (dmis-lab/biobert-base-cased-v1.2, [CLS], 768-d).
입력 텍스트만 다름:
  - smiles    : drug_smiles.json 의 SMILES 문자열
  - rdkitdesc : case3/rdkit_desc.json 의 RDKit 30-feature JSON 문자열 (SMILES 포함)
출력: ddibn/precompute/biobert_smiles.pt , biobert_rdkitdesc.pt
      {'embeddings': {int id: Tensor[768]}, 'hidden_dim':768, 'model_name':..., 'input_type':...}

Run (CPU, GPU 점유 회피): micromamba run -n DDIBench python case3/make_biobert_doc.py
"""
import json, os, time, torch

HERE = os.path.dirname(os.path.abspath(__file__))          # .../ddi334/case3
DDI334 = os.path.dirname(HERE)
PRECOMP = os.path.join(DDI334, "ddibn", "precompute")
MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"
DEVICE = os.environ.get("DOC_DEVICE", "cpu")
MAX_TOKENS = 512


def load_inputs():
    smiles = {int(k): v for k, v in json.load(open(os.path.join(DDI334, "ddibn", "precompute", "drug_smiles.json"))).items()}
    rdkit = {int(k): v for k, v in json.load(open(os.path.join(HERE, "rdkit_desc.json"))).items()}
    return smiles, rdkit


def encode(texts_map, input_type, out_name):
    from transformers import AutoTokenizer, AutoModel
    ids = sorted(texts_map.keys())
    texts = [texts_map[i] for i in ids]
    print(f"[{input_type}] drugs={len(ids)} device={DEVICE}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    hd = model.config.hidden_size
    emb = {}; t0 = time.time(); bs = 16
    with torch.no_grad():
        for i in range(0, len(ids), bs):
            bids = ids[i:i + bs]; bt = texts[i:i + bs]
            t = tok(bt, padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="pt")
            t = {k: v.to(DEVICE) for k, v in t.items()}
            cls = model(**t).last_hidden_state[:, 0, :].detach().clone().cpu().float()
            for did, e in zip(bids, cls):
                emb[int(did)] = e
            if i % (bs * 5) == 0:
                print(f"  [{i + len(bids)}/{len(ids)}] {time.time()-t0:.0f}s", flush=True)
    out = os.path.join(PRECOMP, out_name)
    torch.save({"embeddings": emb, "hidden_dim": hd, "model_name": MODEL_NAME,
                "input_type": input_type, "fallback_count": 0}, out)
    print(f"Saved: {out} ({len(emb)} drugs, dim={hd}, {time.time()-t0:.0f}s)", flush=True)


def main():
    smiles, rdkit = load_inputs()
    encode(smiles, "smiles_c3", "biobert_smiles.pt")
    encode(rdkit, "rdkitdesc_c3", "biobert_rdkitdesc.pt")


if __name__ == "__main__":
    main()
