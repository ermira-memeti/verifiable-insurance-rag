#!/usr/bin/env python3
"""Reproduce the main descriptive, paired, clustered, and abstention analyses."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, binomtest
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "evaluation" / "human_answer_scores_deidentified.csv"
OUT = ROOT / "reproduced_results"
(OUT / "tables").mkdir(parents=True, exist_ok=True)
(OUT / "figures").mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
metrics = ["Correctness","Faithfulness","Completeness","CitationSupport","EvidenceHandling"]

# Overall descriptives.
rows=[]
for method in ["G1","G2"]:
    s=df[df.Generation_Method==method]
    row={"Method":method,"N":len(s)}
    for m in metrics:
        row[f"{m} Mean"]=s[m].mean()
        row[f"{m} Median"]=s[m].median()
        row[f"{m} Perfect Rate"]=(s[m]==2).mean()
    row["Unsupported Claims n"]=(s.Unsupported=="YES").sum()
    row["Contradictions n"]=(s.Contradiction=="YES").sum()
    rows.append(row)
overall=pd.DataFrame(rows)
overall.to_csv(OUT/"tables"/"overall_results.csv",index=False)

# Matched pairs.
pair_rows=[]
for ec,g in df.groupby("Evidence_Case_ID"):
    assert set(g.Generation_Method)=={"G1","G2"}
    row={"Evidence_Case_ID":ec,"Question_ID":g.Question_ID.iloc[0],"Condition_ID":g.Condition_ID.iloc[0],"Evidence_State":g.Final_Human_Evidence_Sufficiency.iloc[0]}
    for _,x in g.iterrows():
        method=x.Generation_Method
        for m in metrics:
            row[f"{m}_{method}"]=x[m]
        row[f"Abstained_{method}"]=int(bool(x.Model_Abstained))
        expected=x.Final_Human_Evidence_Sufficiency=="INSUFFICIENT"
        row[f"AbstentionAligned_{method}"]=int(bool(x.Model_Abstained)==expected)
    pair_rows.append(row)
paired=pd.DataFrame(pair_rows)
paired.to_csv(OUT/"tables"/"paired_evidence_cases.csv",index=False)

# Scenario-level and question-level paired sensitivity plus cluster bootstrap.
rng=np.random.default_rng(20261103)
test_rows=[]
for m in metrics:
    d=paired[f"{m}_G2"]-paired[f"{m}_G1"]
    nz=d[d!=0]
    scenario_sign=float(binomtest(int((nz>0).sum()),len(nz),.5).pvalue) if len(nz) else np.nan
    scenario_wil=float(wilcoxon(paired[f"{m}_G2"],paired[f"{m}_G1"],zero_method="wilcox").pvalue) if len(nz) else np.nan
    q=paired.groupby("Question_ID")[[f"{m}_G1",f"{m}_G2"]].mean()
    qd=q[f"{m}_G2"]-q[f"{m}_G1"]
    qnz=qd[qd!=0]
    question_sign=float(binomtest(int((qnz>0).sum()),len(qnz),.5).pvalue) if len(qnz) else np.nan
    question_wil=float(wilcoxon(q[f"{m}_G2"],q[f"{m}_G1"],zero_method="wilcox").pvalue) if len(qnz) else np.nan
    qdiff=(q[f"{m}_G2"]-q[f"{m}_G1"]).to_numpy(dtype=float)
    idx=rng.integers(0,len(qdiff),size=(20000,len(qdiff)))
    boots=qdiff[idx].mean(axis=1)
    lo,hi=np.quantile(boots,[.025,.975])
    test_rows.append({
        "Measure":m,"G2 Better":int((d>0).sum()),"G1 Better":int((d<0).sum()),"Ties":int((d==0).sum()),
        "Scenario Sign p":scenario_sign,"Scenario Wilcoxon p":scenario_wil,
        "Question G2 Better":int((qd>0).sum()),"Question G1 Better":int((qd<0).sum()),"Question Ties":int((qd==0).sum()),
        "Question Sign p":question_sign,"Question Wilcoxon p":question_wil,
        "Cluster CI Lower":lo,"Cluster CI Upper":hi
    })
pd.DataFrame(test_rows).to_csv(OUT/"tables"/"paired_tests.csv",index=False)

# Abstention alignment.
a=paired.AbstentionAligned_G1; b=paired.AbstentionAligned_G2
bbetter=int(((a==0)&(b==1)).sum()); abetter=int(((a==1)&(b==0)).sum())
case_p=float(binomtest(min(bbetter,abetter),bbetter+abetter,.5).pvalue) if bbetter+abetter else 1.0
q=paired.groupby("Question_ID")[["AbstentionAligned_G1","AbstentionAligned_G2"]].mean()
qd=q.AbstentionAligned_G2-q.AbstentionAligned_G1
qnz=qd[qd!=0]
qsign=float(binomtest(int((qnz>0).sum()),len(qnz),.5).pvalue) if len(qnz) else np.nan
qwil=float(wilcoxon(q.AbstentionAligned_G2,q.AbstentionAligned_G1,zero_method="wilcox").pvalue) if len(qnz) else np.nan
qdiff=(q.AbstentionAligned_G2-q.AbstentionAligned_G1).to_numpy(dtype=float)
idx=rng.integers(0,len(qdiff),size=(20000,len(qdiff)))
boots=qdiff[idx].mean(axis=1)
lo,hi=np.quantile(boots,[.025,.975])
pd.DataFrame([{
    "G1 Alignment":a.mean(),"G2 Alignment":b.mean(),"Difference":b.mean()-a.mean(),
    "G2 Better Discordant":bbetter,"G1 Better Discordant":abetter,"McNemar Exact p":case_p,
    "Question Sign p":qsign,"Question Wilcoxon p":qwil,"Cluster CI Lower":lo,"Cluster CI Upper":hi
}]).to_csv(OUT/"tables"/"abstention_tests.csv",index=False)

# Figures.
plot=overall.set_index("Method")[[f"{m} Mean" for m in metrics]].T
plot.index=[m.replace("CitationSupport","Citation support").replace("EvidenceHandling","Evidence handling") for m in metrics]
ax=plot.plot(kind="bar",figsize=(9,5)); ax.set_ylim(0,2.05); ax.set_ylabel("Mean human score (0-2)"); ax.set_title("Overall human-evaluated answer quality"); plt.xticks(rotation=30,ha="right"); plt.tight_layout(); plt.savefig(OUT/"figures"/"overall_human_scores.png",dpi=220); plt.close()

state=[]
for evidence_state in ["SUFFICIENT","PARTIAL","INSUFFICIENT"]:
    for method in ["G1","G2"]:
        s=df[(df.Final_Human_Evidence_Sufficiency==evidence_state)&(df.Generation_Method==method)]
        state.append({"Evidence State":evidence_state,"Method":method,"Formal Abstention Rate":s.Model_Abstained.astype(bool).mean()})
state=pd.DataFrame(state)
state.to_csv(OUT/"tables"/"abstention_by_evidence_state.csv",index=False)
pivot=state.pivot(index="Evidence State",columns="Method",values="Formal Abstention Rate").loc[["SUFFICIENT","PARTIAL","INSUFFICIENT"]]*100
ax=pivot.plot(kind="bar",figsize=(8,5)); ax.set_ylabel("Formal abstention rate (%)"); ax.set_title("Formal abstention by human-validated evidence sufficiency"); plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(OUT/"figures"/"abstention_by_evidence_state.png",dpi=220); plt.close()

print(f"Wrote reproduced outputs to {OUT}")
