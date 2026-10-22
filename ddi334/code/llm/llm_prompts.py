#!/usr/bin/env python3
"""DDI-334 LLM 프롬프트 빌더 (V2 binary-with-R).

02 §3.1 설계 그대로:
  - 입력: Name + SMILES (DOC LLM 셀 양식)
  - 후보: 그 행의 queried side effect(vec=1)만 (전 실험과 동일하게 queried만 사용)
  - side effect 1개 질문 -> Yes/No (guided_choice + logprob)
  - type 이름: data/splits/typenames.json
"""
import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPLITS_DIR = os.path.join(HERE, 'data', 'splits')
MOL_DIR = os.path.join(HERE, 'data', 'molecule')
DOC_DIR = os.path.join(HERE, 'data', 'document')
NAMES_JSON = os.path.join(SPLITS_DIR, 'drug_names.json')      # {db_id: name} (333/334; db_id 325엔 이름 소스 자체가 없어 drug_{id} fallback)
SMILES_JSON = os.path.join(MOL_DIR, 'drug_smiles.json')       # {db_id: smiles}, 334 전부

V2_SYSTEM = (
    "You are a pharmacology expert. Given information on two drugs and a question\n"
    "about a side effect, answer with Yes or No.\n"
    "Output your answer after ##Answer: <Yes or No>"
)


class DDI334Prompt:
    def __init__(self):
        self.typenames = json.load(open(os.path.join(SPLITS_DIR, 'typenames.json')))
        self.name_map = {int(k): v for k, v in json.load(open(NAMES_JSON)).items()}
        self.smiles = {int(k): v for k, v in json.load(open(SMILES_JSON)).items()}
        # bio profile (Existing drug 입력): db_id -> {genes, indications, pharm_class}  (build_bio_profile_doc.py)
        bp_path = os.path.join(DOC_DIR, 'bio_profile_text.json')
        self.bio = json.load(open(bp_path)) if os.path.exists(bp_path) else {}

    def name(self, db):
        return self.name_map.get(db) or f"drug_{db}"

    # ── 약물 1개 템플릿 텍스트 (tier: novel_drug/existing_drug) — 접두어("Drug N") 없음.
    #    LLM 프롬프트 블록(_block_novel_drug/_block_existing_drug)과 cell12 BioBERT 텍스트(build_biobert_template.py)가
    #    이 함수 하나를 공유한다 (05_실험설계(DDI334).md §8, 2026-07-27 통일).
    def drug_text(self, d, tier='existing_drug'):
        """tier='novel_drug': SMILES만(이름 없음, 완전 신약 가정).
        tier='existing_drug' : + Name + Target genes + Indications + Pharmacologic class.
        모든 약물이 동일 포맷 유지 — 정보 없으면 '(none)'으로 명시."""
        if tier == 'novel_drug':
            return f"SMILES: {self.smiles.get(d,'')}\n"
        bp = self.bio.get(str(d), {})
        g = ', '.join(bp.get('genes') or []) or "(none)"
        text = f"Name: {self.name(d)}\nSMILES: {self.smiles.get(d,'')}\nTarget genes: {g}\n"
        ind = ', '.join(bp.get('indications') or []) or "(none)"
        cl = ', '.join(bp.get('pharm_class') or []) or "(none)"
        return text + f"Indications: {ind}\nPharmacologic class: {cl}\n"

    def _indent_block(self, i, text):
        body = "\n".join("  " + line for line in text.rstrip("\n").split("\n"))
        return f"Drug {i}\n{body}\n"

    # ── Existing/Novel 약물 블록 (프롬프트용 "Drug N" 접두 + 2-space indent) ──
    def _block_novel_drug(self, i, d):
        """Novel drug: 구조만."""
        return self._indent_block(i, self.drug_text(d, 'novel_drug'))

    def _block_existing_drug(self, i, d):
        """Existing drug: name + SMILES + target genes(binds) + indications + pharmacologic class."""
        return self._indent_block(i, self.drug_text(d, 'existing_drug'))

    # ── V2: binary-with-R, one side effect.
    #   mode='existing_drug'(name+SMILES+genes+ind+class) / 'novel_drug'(SMILES만) ──
    def build_v2(self, d1, d2, type_id, mode='existing_drug'):
        blk = self._block_novel_drug if mode == 'novel_drug' else self._block_existing_drug
        # 질문은 익명(Drug 1/Drug 2) — Novel은 이름이 없으니 일관되게 양쪽 동일
        user = (blk(1, d1) + "\n" + blk(2, d2) + "\n"
                + f"Question: Does co-administration of Drug 1 and Drug 2 cause "
                + f"{self.typenames[str(type_id)]}?\n\n##Answer:")
        return V2_SYSTEM, user


RENDER_SPLITS = ["train", "valid_S0", "test_S0", "valid_S1", "test_S1", "valid_S2", "test_S2"]


def render_cache():
    """V2 Existing/Novel 프롬프트를 미리 렌더 -> data/prompt/{existing_drug,novel_drug}/{split}.jsonl.
    각 줄: {"d1","d2","type","y","user"} (y=polarity: 1 양성쌍 / 0 음성쌍). build_v2(mode=...)와
    동일 로직 -> 재생성 가능(gitignore 대상). 입력 데이터(data/splits, data/document 등)는 그대로
    유지, 산출 프롬프트만 data/prompt/에. (render_prompts.py 흡수, 2026-10-20)"""
    pb = DDI334Prompt()
    out = os.path.join(HERE, "data", "prompt")
    MODES = ("existing_drug", "novel_drug")
    for m in MODES:
        os.makedirs(os.path.join(out, m), exist_ok=True)
    total = 0
    for sp in RENDER_SPLITS:
        path = os.path.join(SPLITS_DIR, f"{sp}.txt")
        if not os.path.exists(path):
            continue
        fh = {m: open(os.path.join(out, m, f"{sp}.jsonl"), "w") for m in MODES}
        n = 0
        for line in open(path):
            p = line.split()
            if len(p) < 4:
                continue
            d1, d2, pol = int(p[0]), int(p[1]), int(p[3])
            qids = [i for i, x in enumerate(p[2].split(',')) if x == '1']
            for t in qids:
                for m in MODES:
                    _, u = pb.build_v2(d1, d2, t, mode=m)
                    fh[m].write(json.dumps({"d1": d1, "d2": d2, "type": t, "y": pol, "user": u}, ensure_ascii=False) + "\n")
                n += 1
        for f in fh.values():
            f.close()
        total += n
        print(f"  {sp}: {n} prompts", flush=True)
    print(f"[render] 총 {total} (쌍,타입) x existing_drug/novel_drug -> {out}/")


def demo():
    pb = DDI334Prompt()
    # 실제 첫 positive 행에서 queried type 뽑아 프롬프트 시연
    for line in open(os.path.join(SPLITS_DIR, 'test_S0.txt')):
        p = line.split()
        if p[3] == '1':
            d1, d2 = int(p[0]), int(p[1])
            qids = [i for i, x in enumerate(p[2].split(',')) if x == '1'][:5]  # 처음 5개만 시연
            s2, u2 = pb.build_v2(d1, d2, qids[0])
            print("="*70, "\n[System]\n", s2, "\n\n[User]\n", u2)
            break


if __name__ == '__main__':
    # demo (기본):  python llm_prompts.py
    # render cache: python llm_prompts.py render
    import sys
    args = sys.argv[1:]
    if args and args[0] == 'render':
        render_cache()
    else:
        demo()
