#!/usr/bin/env python3
"""Reproduce the exploratory retrieval-to-evidence-sufficiency analysis."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]
runs=pd.read_csv(ROOT/'data/retrieval/retrieval_runs.csv')
ev=pd.read_csv(ROOT/'data/core/evidence_map.csv')
ans=pd.read_csv(ROOT/'data/evaluation/human_answer_scores_deidentified.csv')
OUT=ROOT/'reproduced_results'
(OUT/'tables').mkdir(parents=True,exist_ok=True)
(OUT/'figures').mkdir(parents=True,exist_ok=True)

c3=ans[ans.Condition_ID=='C3_RETRIEVED'].copy()
rows=[]
for qid in sorted(c3.Question_ID.unique()):
    gold=ev[(ev.question_id==qid)&(ev.required_flag==True)]
    units=gold.passage_id.tolist(); distinct=list(dict.fromkeys(units))
    rr=runs[(runs.question_id==qid)&(runs.method_id=='Full-IKO-V2')&(runs['rank']<=5)].sort_values('rank')
    top=rr.passage_id.tolist(); top_set=set(top)
    ranks=[i+1 for i,p in enumerate(top) if p in set(distinct)]
    row={
        'Question_ID':qid,
        'Hit5':int(any(p in top_set for p in distinct)),
        'MRR5':1/min(ranks) if ranks else 0.0,
        'CompleteEvidence5':int(set(distinct).issubset(top_set)),
        'ECS5':sum(p in top_set for p in units)/len(units),
        'HumanEvidenceState':c3[c3.Question_ID==qid].Final_Human_Evidence_Sufficiency.iloc[0]
    }
    rows.append(row)
retr=pd.DataFrame(rows)
state_order={'INSUFFICIENT':0,'PARTIAL':1,'SUFFICIENT':2}
retr['HumanEvidenceStateOrdinal']=retr.HumanEvidenceState.map(state_order)
retr.to_csv(OUT/'tables'/'retrieval_to_answer_question_level.csv',index=False)

corr=[]
for m in ['Hit5','MRR5','CompleteEvidence5','ECS5']:
    rho,p=spearmanr(retr[m],retr.HumanEvidenceStateOrdinal)
    corr.append({'Metric':m,'Spearman rho':rho,'p-value':p,'N':len(retr)})
pd.DataFrame(corr).to_csv(OUT/'tables'/'retrieval_semantic_sufficiency_correlations.csv',index=False)

jitter=np.linspace(-.045,.045,len(retr))
y=retr.HumanEvidenceStateOrdinal+jitter
plt.figure(figsize=(8,5)); plt.scatter(retr.ECS5,y)
plt.yticks([0,1,2],['Insufficient','Partial','Sufficient'])
plt.xlabel('Evidence Coverage Score at 5 (ECS@5)'); plt.ylabel('Human-validated semantic evidence state')
plt.title('Retrieved evidence coverage vs semantic evidence sufficiency (n=12)'); plt.tight_layout()
plt.savefig(OUT/'figures'/'ecs_vs_semantic_sufficiency.png',dpi=220); plt.close()
print(f"Wrote retrieval analysis to {OUT}")
