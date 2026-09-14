#!/usr/bin/env python3
"""DDI-334 LLM 프롬프트 빌더 (V1 multi-label JSON / V2 binary-with-R).

02 §3.1 설계 그대로:
  - 입력: Name + SMILES (DOC LLM 셀 양식)
  - 후보: 그 행의 queried side effect(vec=1)만 (전 실험과 동일하게 queried만 사용)
  - V1: candidate side effects 목록 -> JSON {id: true/false} (guided_json)
  - V2: side effect 1개 질문 -> Yes/No (guided_choice + logprob)
  - type 이름: twosides_typenames_{ds}.json
"""
import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.dirname(HERE)
LATEX = os.path.abspath(os.path.join(EXP, '..', '..', '..'))
NAMES_JSON = os.path.join(LATEX, 'data', 'embeddings', 'drugbank_names.json')
SMILES_JSON = os.path.join(LATEX, 'data', 'smiles', 'drugbank_id2smiles.json')

V1_SYSTEM = (
    "You are a pharmacology expert. Given information on two drugs and a list of\n"
    "candidate side effects, decide for EACH listed side effect whether\n"
    "co-administration of the two drugs causes it.\n"
    "Output only a JSON object mapping each side effect id (as a string) to\n"
    "true or false after ##Answer:"
)
V2_SYSTEM = (
    "You are a pharmacology expert. Given information on two drugs and a question\n"
    "about a side effect, answer with Yes or No.\n"
    "Output your answer after ##Answer: <Yes or No>"
)


class DDI334Prompt:
    def __init__(self, dataset):
        self.typenames = json.load(open(os.path.join(HERE, 'meta', f'twosides_typenames_{dataset}.json')))
        names_raw = json.load(open(NAMES_JSON))['drug2name']
        self.name_map = {int(k): v['name'] for k, v in names_raw.items()}
        self.smiles = {int(k): v for k, v in json.load(open(SMILES_JSON)).items()}
        # bio profile (Ideal 입력): db_id -> {genes, indications, pharm_class}  (build_bio_profile.py)
        bp_path = os.path.join(HERE, 'meta', 'bio_profile_llm.json')
        self.bio = json.load(open(bp_path)) if os.path.exists(bp_path) else {}

    def name(self, db):
        return self.name_map.get(db) or f"drug_{db}"

    def _drug_block(self, d1, d2):
        return (f"Drug 1\n  Name: {self.name(d1)}\n  SMILES: {self.smiles.get(d1,'')}\n\n"
                f"Drug 2\n  Name: {self.name(d2)}\n  SMILES: {self.smiles.get(d2,'')}\n")

    # ── V1: multi-label JSON over queried type ids ──
    def build_v1(self, d1, d2, type_ids):
        cand = "\n".join(f"{t}: {self.typenames[str(t)]}" for t in type_ids)
        user = (self._drug_block(d1, d2) + "\nCandidate side effects:\n" + cand + "\n\n##Answer:")
        schema = {"type": "object",
                  "properties": {str(t): {"type": "boolean"} for t in type_ids},
                  "required": [str(t) for t in type_ids],
                  "additionalProperties": False}
        return V1_SYSTEM, user, schema

    # ── 약물 1개 템플릿 텍스트 (tier: real/genes/ideal) — 접두어("Drug N") 없음.
    #    LLM 프롬프트 블록(_block_real/_block_ideal)과 cell13 BioBERT 텍스트(build_biobert_template.py)가
    #    이 함수 하나를 공유한다 (05_실험설계(DDI334).md §8, 2026-07-27 통일).
    def drug_text(self, d, tier='ideal'):
        """tier='real': SMILES만(이름 없음, 완전 신약 가정).
        tier='genes' : + Name + Target genes.
        tier='ideal' : + Indications + Pharmacologic class.
        모든 약물이 동일 포맷 유지 — 정보 없으면 '(none)'으로 명시."""
        if tier == 'real':
            return f"SMILES: {self.smiles.get(d,'')}\n"
        bp = self.bio.get(str(d), {})
        g = ', '.join(bp.get('genes') or []) or "(none)"
        text = f"Name: {self.name(d)}\nSMILES: {self.smiles.get(d,'')}\nTarget genes: {g}\n"
        if tier == 'genes':
            return text
        ind = ', '.join(bp.get('indications') or []) or "(none)"
        cl = ', '.join(bp.get('pharm_class') or []) or "(none)"
        return text + f"Indications: {ind}\nPharmacologic class: {cl}\n"

    # ── 자연어 버전 (cell13 DOC 추가 실험, 2026-07-28) — drug_text()와 동일 정보,
    #    key:value 템플릿 대신 문장으로 서술. 정보 없으면 "no known ~ documented"로 명시(생략 안 함).
    def drug_text_nl(self, d, tier='ideal'):
        if tier == 'real':
            return f"The molecule has the SMILES structure {self.smiles.get(d,'')}."
        bp = self.bio.get(str(d), {})
        name, smi = self.name(d), self.smiles.get(d, '')
        genes = bp.get('genes') or []
        if genes:
            sent = f"{name}, with the chemical structure {smi}, is known to interact with the following genes: {', '.join(genes)}."
        else:
            sent = f"{name}, with the chemical structure {smi}, has no known target gene interactions documented."
        if tier == 'genes':
            return sent
        ind = bp.get('indications') or []
        cl = bp.get('pharm_class') or []
        sent += (f" It is indicated for {', '.join(ind)}." if ind
                 else " No known indications are documented for this drug.")
        sent += (f" It is classified as a {', '.join(cl)}." if cl
                 else " No known pharmacologic class is documented for this drug.")
        return sent

    def _indent_block(self, i, text):
        body = "\n".join("  " + line for line in text.rstrip("\n").split("\n"))
        return f"Drug {i}\n{body}\n"

    # ── Ideal/Real 약물 블록 (프롬프트용 "Drug N" 접두 + 2-space indent) ──
    def _block_real(self, i, d):
        """Real(신약): 구조만."""
        return self._indent_block(i, self.drug_text(d, 'real'))

    def _block_ideal(self, i, d, genes_only=False):
        """Ideal: name + SMILES + target genes(binds) [+ indications + pharmacologic class].
        genes_only=True 면 indications/class 제외 (Ideal 내부 ablation: genes만)."""
        return self._indent_block(i, self.drug_text(d, 'genes' if genes_only else 'ideal'))

    # ── V2: binary-with-R, one side effect.
    #   mode='ideal'(name+SMILES+genes+ind+class) / 'ideal_genes'(name+SMILES+genes) / 'real'(SMILES만) ──
    def build_v2(self, d1, d2, type_id, mode='ideal'):
        if mode == 'real':
            blk = self._block_real
        elif mode == 'ideal_genes':
            blk = lambda i, d: self._block_ideal(i, d, genes_only=True)
        else:
            blk = self._block_ideal
        # 질문은 익명(Drug 1/Drug 2) — Real은 이름이 없으니 일관되게 양쪽 동일
        user = (blk(1, d1) + "\n" + blk(2, d2) + "\n"
                + f"Question: Does co-administration of Drug 1 and Drug 2 cause "
                + f"{self.typenames[str(type_id)]}?\n\n##Answer:")
        return V2_SYSTEM, user


if __name__ == '__main__':
    import sys
    ds = sys.argv[1] if len(sys.argv) > 1 else 'ddibn'
    pb = DDI334Prompt(ds)
    # 실제 첫 positive 행에서 queried type 뽑아 프롬프트 시연
    for line in open(os.path.join(HERE, 'data', ds, 'test_S0.txt')):
        p = line.split()
        if p[3] == '1':
            d1, d2 = int(p[0]), int(p[1])
            qids = [i for i, x in enumerate(p[2].split(',')) if x == '1'][:5]  # 처음 5개만 시연
            s1, u1, sc = pb.build_v1(d1, d2, qids)
            print("="*70, "\nV1 [System]\n", s1, "\n\nV1 [User]\n", u1)
            print("\nV1 guided_json schema:\n", json.dumps(sc, ensure_ascii=False))
            s2, u2 = pb.build_v2(d1, d2, qids[0])
            print("="*70, "\nV2 [System]\n", s2, "\n\nV2 [User]\n", u2)
            break
