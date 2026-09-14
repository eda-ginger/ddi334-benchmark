"""cell13 DOC 템플릿 [CLS] 임베딩 3종 (real/genes/ideal) — backbone 선택 가능.

2026-07-27 미팅 피드백 반영 — LLM(§5, llm_prompts.py) real/genes/ideal 템플릿과
동일한 텍스트를 cell13(frozen text encoder)에도 사용 (05_실험설계(DDI334).md §8).
텍스트는 llm_prompts.py의 DDI334Prompt.drug_text()를 그대로 재사용(단일 소스).
2026-07-27 확장(§8.5): DOC 계열 BERT 3종(biobert/pubmedbert/scibert) 비교를 위해
backbone을 --tag로 선택 가능하게 일반화. hidden_dim은 각 모델 config에서 동적으로
읽음(하드코딩 금지 — 모델마다 dimension이 다를 수 있음).

  real  : SMILES만
  genes : + Name + Target genes
  ideal : + Indications + Pharmacologic class

대상 약물 = bio_profile_llm.json의 334드럭(DDI-334 전체, db_id 공간).
전처리 방식은 기존 precompute/scripts/biobert.py와 동일([CLS], max_length=512).

출력: ddi334/data/{ddibn,tdc}/precompute/{tag}_{real,genes,ideal}.pt
      + experiment/data/precompute/{tag}_{real,genes,ideal}.pt (공유 소스본, assemble_precompute.py 관례)
      {'embeddings': {int db_id: Tensor[dim]}, 'hidden_dim':dim(모델별 실제값), 'model_name':..., 'input_type':...}
      (train.py는 실행 시 PRECOMPUTE_DIR을 ddi334/data/{dataset}/precompute/로 재설정하므로
       load_biobert_embeddings가 실제로 읽는 곳은 데이터셋별 경로 — assemble_precompute.py 참조)

Run: micromamba run -n DDIBench python build_biobert_template.py --tag biobert [--device cuda:0]
     micromamba run -n DDIBench python build_biobert_template.py --tag pubmedbert
     micromamba run -n DDIBench python build_biobert_template.py --tag scibert
"""
import argparse
import json
import os
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))                       # ddi334/code
DDI334_DIR = os.path.dirname(HERE)                                      # ddi334
EXPERIMENT_DIR = os.path.dirname(DDI334_DIR)                            # v5_ddi-bench/experiment
SHARED_PRECOMPUTE_DIR = os.path.join(EXPERIMENT_DIR, 'data', 'precompute')          # 공유 소스본
OUT_DIRS = [SHARED_PRECOMPUTE_DIR] + [
    os.path.join(DDI334_DIR, 'data', ds, 'precompute') for ds in ('ddibn', 'tdc')   # 실제 train.py가 읽는 곳
]
BIO_PROFILE = os.path.join(DDI334_DIR, 'meta', 'bio_profile_llm.json')
MAX_TOKENS = 512
TIERS = ['real', 'genes', 'ideal']

# DOC 계열 BERT 3종 (2026-07-27 미팅 논의, 우리 85편 서베이에서 DOC 모달리티 최다 등장 BERT backbone)
MODEL_PRESETS = {
    'biobert': 'dmis-lab/biobert-base-cased-v1.2',
    'pubmedbert': 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext',
    'scibert': 'allenai/scibert_scivocab_uncased',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='biobert', choices=list(MODEL_PRESETS.keys()),
                     help='출력 파일 접두어({tag}_{tier}.pt) 겸 프리셋 모델 선택')
    ap.add_argument('--model_name', default=None, help='프리셋 대신 임의 HF 모델 지정(선택)')
    ap.add_argument('--style', default='template', choices=['template', 'natural'],
                     help='template=drug_text() key:value / natural=drug_text_nl() 문장체 (2026-07-28 추가 실험)')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--batch_size', type=int, default=16)
    args = ap.parse_args()
    model_name = args.model_name or MODEL_PRESETS[args.tag]

    import sys
    sys.path.insert(0, HERE)
    from llm_prompts import DDI334Prompt

    pb = DDI334Prompt('ddibn')   # typenames는 drug_text에 안 쓰이므로 dataset 무관
    drug_ids = sorted(int(k) for k in json.load(open(BIO_PROFILE)).keys())
    print(f'Target drugs: {len(drug_ids)}')

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Loading {model_name} on {device}...')
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    hidden_dim = model.config.hidden_size   # 모델별 실제 dimension (하드코딩 금지)
    print(f'  hidden_dim={hidden_dim}')

    text_fn = pb.drug_text if args.style == 'template' else pb.drug_text_nl
    tag_out = args.tag if args.style == 'template' else f'{args.tag}_nl'
    for tier in TIERS:
        texts = [text_fn(d, tier) for d in drug_ids]
        embeddings = {}
        t0 = time.time()
        n = len(drug_ids)
        with torch.no_grad():
            for i in range(0, n, args.batch_size):
                batch_ids = drug_ids[i:i + args.batch_size]
                batch_texts = texts[i:i + args.batch_size]
                tok = tokenizer(batch_texts, padding=True, truncation=True,
                                 max_length=MAX_TOKENS, return_tensors='pt')
                tok = {k: v.to(device) for k, v in tok.items()}
                out = model(**tok)
                cls = out.last_hidden_state[:, 0, :].detach().clone().cpu().float()
                for did, emb in zip(batch_ids, cls):
                    embeddings[int(did)] = emb
            print(f'  [{tier}] {n}/{n} wall={time.time()-t0:.0f}s', flush=True)

        payload = {
            'embeddings': embeddings,
            'hidden_dim': hidden_dim,
            'model_name': model_name,
            'input_type': tier,
            'fallback_count': 0,
        }
        for od in OUT_DIRS:
            os.makedirs(od, exist_ok=True)
            out_pt = os.path.join(od, f'{tag_out}_{tier}.pt')
            torch.save(payload, out_pt)
            print(f'Saved: {out_pt} ({len(embeddings)} drugs, dim={hidden_dim})')


if __name__ == '__main__':
    main()
