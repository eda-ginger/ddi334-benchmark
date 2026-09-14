#!/usr/bin/env python3
"""SMILES -> RDKit 30-feature structure description (Case-3 DOC 입력).
모두 SMILES에서 RDKit 직접계산 (외부DB·실측 0 -> Case-3 유효). 근거: 09 문서 §4.1.
A.필수6(Lipinski+Veber) B.물리화학4 C.원소조성5 D.고리/방향족3 E.입체/이온화3 F.작용기9 = 30.
출력: case3/rdkit_desc.json {drug_id: description}."""
import json, os
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, rdMolDescriptors, Fragments, QED, GraphDescriptors

HERE = os.path.dirname(os.path.abspath(__file__))
SM = json.load(open(os.path.join(HERE, "..", "ddibn", "precompute", "drug_smiles.json")))
HALO = {9, 17, 35, 53, 85}
ACID_SMARTS = [Chem.MolFromSmarts(s) for s in ["[CX3](=O)[OX2H1]", "[SX4](=O)(=O)[OX2H1]"]]
BASE_SMARTS = Chem.MolFromSmarts("[NX3;!$(NC=O);!$(N=O);!$(NS(=O)=O);!$(N#*)]")
FG = [("carboxylic acid", Fragments.fr_COO), ("ester", Fragments.fr_ester),
      ("amide", Fragments.fr_amide), ("tertiary amine", Fragments.fr_NH0),
      ("ether", Fragments.fr_ether), ("hydroxyl", lambda m: Fragments.fr_Al_OH(m)+Fragments.fr_Ar_OH(m)),
      ("nitro", Fragments.fr_nitro), ("sulfonamide", Fragments.fr_sulfonamd),
      ("aromatic amine", Fragments.fr_aniline)]

def desc(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None: return None
    # A 필수6
    mw=Descriptors.MolWt(m); logp=Crippen.MolLogP(m); hbd=Lipinski.NumHDonors(m)
    hba=Lipinski.NumHAcceptors(m); rot=Descriptors.NumRotatableBonds(m); tpsa=Descriptors.TPSA(m)
    # B 물리화학4
    qed=QED.qed(m); mr=Crippen.MolMR(m); fsp3=Descriptors.FractionCSP3(m); bertz=GraphDescriptors.BertzCT(m)
    # C 원소조성5
    heavy=m.GetNumHeavyAtoms(); het=rdMolDescriptors.CalcNumHeteroatoms(m)
    nN=sum(a.GetSymbol()=='N' for a in m.GetAtoms()); nO=sum(a.GetSymbol()=='O' for a in m.GetAtoms())
    nHal=sum(a.GetAtomicNum() in HALO for a in m.GetAtoms())
    # D 고리/방향족3
    rings=rdMolDescriptors.CalcNumRings(m); arom=rdMolDescriptors.CalcNumAromaticRings(m)
    aromat=sum(a.GetIsAromatic() for a in m.GetAtoms())
    # E 입체/이온화3
    stereo=len(Chem.FindMolChiralCenters(m, includeUnassigned=True, useLegacyImplementation=False))
    acid=sum(len(m.GetSubstructMatches(s)) for s in ACID_SMARTS)
    base=len(m.GetSubstructMatches(BASE_SMARTS))
    # F 작용기9
    fgs=[name for name,fn in FG if fn(m)>0]
    d={"SMILES":smiles,"MW":round(mw,1),"logP":round(logp,2),"HBD":hbd,"HBA":hba,"rotatable_bonds":rot,"TPSA":round(tpsa,1),
       "QED":round(qed,2),"MolMR":round(mr,1),"fraction_sp3":round(fsp3,2),"BertzCT":round(bertz),
       "heavy_atoms":heavy,"heteroatoms":het,"N":nN,"O":nO,"halogens":nHal,
       "rings":rings,"aromatic_rings":arom,"aromatic_atoms":aromat,
       "stereocenters":stereo,"acidic_groups":acid,"basic_groups":base,
       "functional_groups":fgs}
    return json.dumps(d, ensure_ascii=False)

out={}; fail=0
for did,smi in SM.items():
    d=desc(smi)
    if d is None: fail+=1
    else: out[did]=d
json.dump(out, open(os.path.join(HERE,"rdkit_desc.json"),"w"), indent=1)
print(f"생성 {len(out)} / 실패 {fail} (총 {len(SM)})")
for did in list(out)[:2]:
    print(f"\n[{did}]\n  {out[did]}")
