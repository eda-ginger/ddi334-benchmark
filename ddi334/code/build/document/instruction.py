#!/usr/bin/env python3
"""DDI-334 Instruction (Cell 13-18, LLM prompt builder, V2 binary-with-R).

Table 1/Table 2's "Instruction" row — the prompt-construction library used
by llm/zeroshot.py (zero-shot), llm/finetuning_train.py and llm/finetuning_eval.py
(fine-tuning). Lives in build/document/ alongside mention_description.py
(the two share one text source, data/document/bio_profile_text.json) even
though, unlike every other build/ script, this one isn't a run-once
precompute step: build_v2() is called live by its llm/ consumers on every
inference, and render_cache() below is only an optional jsonl pre-render
for finetuning_train.py's convenience.

Follows the 02 section 3.1 design as-is:
  - Input: Name + SMILES (Document LLM cell format)
  - Candidates: only that row's queried side effect (vec=1) (only queried, same as every earlier experiment)
  - One side-effect question -> Yes/No (guided_choice + logprob)
  - Type names: data/splits/typenames.json

Run (optional prompt-cache pre-render, not required): python build/document/instruction.py render
"""
import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SPLITS_DIR = os.path.join(HERE, 'data', 'splits')
MOL_DIR = os.path.join(HERE, 'data', 'molecule')
DOC_DIR = os.path.join(HERE, 'data', 'document')
NAMES_JSON = os.path.join(SPLITS_DIR, 'drug_names.json')      # {db_id: name} (333/334; db_id 325 has no name source at all, so it falls back to drug_{id})
SMILES_JSON = os.path.join(MOL_DIR, 'drug_smiles.json')       # {db_id: smiles}, all 334

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
        # bio profile (Existing drug input): db_id -> {genes, indications, pharm_class}  (mention_description.py)
        bp_path = os.path.join(DOC_DIR, 'bio_profile_text.json')
        self.bio = json.load(open(bp_path)) if os.path.exists(bp_path) else {}

    def name(self, db):
        return self.name_map.get(db) or f"drug_{db}"

    # -- Single-drug template text (tier: novel_drug/existing_drug) -- no "Drug N" prefix.
    #    Shared by both the LLM prompt blocks (_block_novel_drug/_block_existing_drug) and the
    #    cell12 BERT text (mention_description.py) through this one function.
    def drug_text(self, d, tier='existing_drug'):
        """tier='novel_drug': SMILES only (no name, assumes a fully novel drug).
        tier='existing_drug' : + Name + Target genes + Indications + Pharmacologic class.
        Every drug keeps the same format -- missing info is explicitly shown as '(none)'."""
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

    # -- Existing/Novel drug blocks ("Drug N" prefix for the prompt + 2-space indent) --
    def _block_novel_drug(self, i, d):
        """Novel drug: structure only."""
        return self._indent_block(i, self.drug_text(d, 'novel_drug'))

    def _block_existing_drug(self, i, d):
        """Existing drug: name + SMILES + target genes(binds) + indications + pharmacologic class."""
        return self._indent_block(i, self.drug_text(d, 'existing_drug'))

    # -- V2: binary-with-R, one side effect.
    #   mode='existing_drug' (name+SMILES+genes+ind+class) / 'novel_drug' (SMILES only) --
    def build_v2(self, d1, d2, type_id, mode='existing_drug'):
        blk = self._block_novel_drug if mode == 'novel_drug' else self._block_existing_drug
        # Drugs are referred to anonymously (Drug 1/Drug 2) in the question -- Novel has no
        # name, so both sides stay consistent
        user = (blk(1, d1) + "\n" + blk(2, d2) + "\n"
                + f"Question: Is co-administration of Drug 1 and Drug 2 associated with "
                + f"{self.typenames[str(type_id)]}?\n\n##Answer:")
        return V2_SYSTEM, user


RENDER_SPLITS = ["train", "valid_S0", "test_S0", "valid_S1", "test_S1", "valid_S2", "test_S2"]


def render_cache():
    """Pre-renders the V2 Existing/Novel prompts -> data/prompt/{existing_drug,novel_drug}/{split}.jsonl.
    Each line: {"d1","d2","type","y","user"} (y=polarity: 1 for a positive pair / 0 for a
    negative pair). Same logic as build_v2(mode=...) -> can be regenerated (gitignored).
    Input data (data/splits, data/document, etc.) is left as-is; only the rendered prompts
    go to data/prompt/. (absorbed render_prompts.py, 2026-10-20)"""
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
    print(f"[render] {total} total (pair,type) x existing_drug/novel_drug -> {out}/")


def demo():
    pb = DDI334Prompt()
    # Demonstrate a prompt using the queried type from the first actual positive row
    for line in open(os.path.join(SPLITS_DIR, 'test_S0.txt')):
        p = line.split()
        if p[3] == '1':
            d1, d2 = int(p[0]), int(p[1])
            qids = [i for i, x in enumerate(p[2].split(',')) if x == '1'][:5]  # demo only the first 5
            s2, u2 = pb.build_v2(d1, d2, qids[0])
            print("="*70, "\n[System]\n", s2, "\n\n[User]\n", u2)
            break


if __name__ == '__main__':
    # demo (default):  python build/document/instruction.py
    # render cache: python build/document/instruction.py render
    import sys
    args = sys.argv[1:]
    if args and args[0] == 'render':
        render_cache()
    else:
        demo()
