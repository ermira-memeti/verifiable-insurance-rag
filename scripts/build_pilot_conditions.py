from pathlib import Path
import pandas as pd
import json, hashlib

ROOT = Path(__file__).resolve().parents[1]
Q = ROOT / "data" / "processed" / "questions.csv"
P = ROOT / "data" / "processed" / "passages.csv"
E = ROOT / "data" / "evidence" / "evidence_map.csv"
R = ROOT / "retrieval" / "frozen_runs" / "retrieval_runs.csv"

PILOT_IDS = [
    "HO-DW-Q-MC-02","HO-DW-Q-MC-05","HO-DW-Q-MC-06","HO-DW-Q-MC-08",
    "HO-GP-Q-MC-01","HO-GP-Q-MC-02","HO-GP-Q-MC-05","HO-GP-Q-MC-07",
    "HO-RCBAP-Q-MC-01","HO-RCBAP-Q-MC-02","HO-RCBAP-Q-MC-04","HO-RCBAP-Q-MC-08"
]
CALIBRATION_ID = "HO-DW-Q-MC-01"
EXCLUDED_KNOWN_ADJUDICATION = ["HO-DW-Q-MC-04","HO-GP-Q-MC-08","HO-RCBAP-Q-MC-07"]
RETRIEVAL_METHOD_C3 = "Full-IKO-V2"
K = 5

q=pd.read_csv(Q); p=pd.read_csv(P); e=pd.read_csv(E); r=pd.read_csv(R)
passage=p.set_index("passage_id").to_dict("index")
pilot_q=q[q.question_id.isin(PILOT_IDS)].copy()
assert len(pilot_q)==12
assert pilot_q.policy_id.value_counts().to_dict()=={"DW":4,"GP":4,"RCBAP":4}

def ranked_non_gold(qid,gold_ids,method="Hybrid-RRF",n=2):
    rr=r[(r.question_id==qid)&(r.method_id==method)].sort_values("rank")
    ids=[]
    for pid in rr.passage_id:
        if pid not in gold_ids and pid not in ids:
            ids.append(pid)
        if len(ids)>=n: break
    return ids

records=[]; rows=[]
for _,qr in pilot_q.sort_values(["policy_id","question_id"]).iterrows():
    qid=qr.question_id
    ee=e[(e.question_id==qid)&(e.required_flag==True)].copy().sort_values("evidence_unit_id")
    gold_ids=list(dict.fromkeys(ee.passage_id.tolist()))
    roles={row.passage_id:row.evidence_role for row in ee.itertuples()}
    assert len(gold_ids)>=2
    c3=r[(r.question_id==qid)&(r.method_id==RETRIEVAL_METHOD_C3)&(r["rank"]<=K)].sort_values("rank").passage_id.tolist()
    distract=ranked_non_gold(qid,set(gold_ids),n=1)
    insufficient=ranked_non_gold(qid,set(gold_ids),n=2)
    conditions={
        "C1_COMPLETE":gold_ids,
        "C2_PARTIAL":gold_ids[:-1],
        "C3_RETRIEVED":c3,
        "C4_DISTRACTOR":gold_ids+distract,
        "C5_INSUFFICIENT":insufficient,
    }
    for cid,ids in conditions.items():
        assert all(x in passage for x in ids)
        assert all(passage[x]["policy_id"]==qr.policy_id for x in ids)
        gp=sum(x in set(gold_ids) for x in ids)
        records.append({
            "pilot_version":"PILOT_V2_FROZEN",
            "question_id":qid,"policy_id":qr.policy_id,"policy_name":qr.policy_name,
            "question_text":qr.question_text,"reference_answer":qr.gold_answer,
            "primary_concept":qr.primary_concept,"condition_id":cid,
            "condition_passage_ids":ids,
            "condition_passages":[{
                "passage_id":pid,"pdf_page":int(passage[pid]["pdf_page"]),
                "passage_text":passage[pid]["passage_text"],
                "is_annotated_required":pid in set(gold_ids),
                "evidence_role":roles.get(pid,"")
            } for pid in ids],
            "annotated_required_passage_ids":gold_ids,
            "annotated_required_count":len(gold_ids),
            "annotated_required_present_count":gp,
            "annotation_status":"PROVISIONAL_STAGE6_PENDING_HUMAN_REVIEW",
            "construction_rule":{
                "C1_COMPLETE":"All currently annotated required passage IDs.",
                "C2_PARTIAL":"All currently annotated required passage IDs except the final mapped required passage.",
                "C3_RETRIEVED":f"Frozen {RETRIEVAL_METHOD_C3} top-{K} ranking from Paper 1.",
                "C4_DISTRACTOR":"All currently annotated required passages plus the highest-ranked Hybrid-RRF non-gold passage.",
                "C5_INSUFFICIENT":"Top two Hybrid-RRF non-gold passages; contains no currently annotated required passage."
            }[cid]
        })
        rows.append({"question_id":qid,"policy_id":qr.policy_id,"condition_id":cid,
                     "n_passages":len(ids),"required_total":len(gold_ids),
                     "required_present":gp,"passage_ids":" | ".join(ids)})

pilot_dir=ROOT/"pilot"; manifest_dir=ROOT/"manifests"
pilot_dir.mkdir(exist_ok=True); manifest_dir.mkdir(exist_ok=True)
jsonl=pilot_dir/"pilot_conditions.jsonl"
with jsonl.open("w",encoding="utf-8") as f:
    for rec in records: f.write(json.dumps(rec,ensure_ascii=False)+"\n")
pd.DataFrame(rows).to_csv(pilot_dir/"pilot_conditions_summary.csv",index=False)
pilot_q.sort_values(["policy_id","question_id"]).to_csv(pilot_dir/"pilot_questions.csv",index=False)

sel=[]
for qid in PILOT_IDS:
    qr=q[q.question_id==qid].iloc[0]
    ec=e[(e.question_id==qid)&(e.required_flag==True)].passage_id.nunique()
    sel.append({"question_id":qid,"policy_id":qr.policy_id,"primary_concept":qr.primary_concept,
                "required_passage_count":ec,"selection_status":"INCLUDED_FINAL_EVALUATION_PILOT"})
qr=q[q.question_id==CALIBRATION_ID].iloc[0]
ec=e[(e.question_id==CALIBRATION_ID)&(e.required_flag==True)].passage_id.nunique()
sel.append({"question_id":CALIBRATION_ID,"policy_id":qr.policy_id,"primary_concept":qr.primary_concept,
            "required_passage_count":ec,"selection_status":"EXCLUDED_PROMPT_CALIBRATION"})
for qid in EXCLUDED_KNOWN_ADJUDICATION:
    qr=q[q.question_id==qid].iloc[0]
    ec=e[(e.question_id==qid)&(e.required_flag==True)].passage_id.nunique()
    sel.append({"question_id":qid,"policy_id":qr.policy_id,"primary_concept":qr.primary_concept,
                "required_passage_count":ec,"selection_status":"EXCLUDED_KNOWN_ADJUDICATION_NOTE"})
pd.DataFrame(sel).to_csv(manifest_dir/"pilot_selection_manifest.csv",index=False)

validation={
    "pilot_version":"PILOT_V2_FROZEN","pilot_questions":12,"conditions_per_question":5,
    "total_condition_records":60,"policy_distribution":{"DW":4,"GP":4,"RCBAP":4},
    "c3_retrieval_method":RETRIEVAL_METHOD_C3,"c3_k":K,
    "prompt_calibration_question_excluded":CALIBRATION_ID,
    "replacement_question":"HO-DW-Q-MC-05",
    "known_adjudication_cases_excluded":EXCLUDED_KNOWN_ADJUDICATION,
    "all_conditions_policy_scoped":True,"all_passage_ids_resolve":True,
    "annotation_warning":"Held-out Stage 6 mappings remain provisional and pending human review."
}
for fp in [jsonl,pilot_dir/"pilot_conditions_summary.csv",pilot_dir/"pilot_questions.csv"]:
    validation["sha256_"+fp.name]=hashlib.sha256(fp.read_bytes()).hexdigest()
(manifest_dir/"pilot_validation.json").write_text(json.dumps(validation,indent=2),encoding="utf-8")
print(json.dumps(validation,indent=2))
