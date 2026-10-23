"""Merge completed fair_reference runs only; keep old/imported records separate."""
import csv, json
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EXPECTED={'batch_size':1,'warmup_images':10,'precision':'fp32','amp':False,'compile':False,'quantization':False,'tensor_rt':False,'rankable_across_models':True}
EXPECTED_MODELS={'mediapipe_pose_heavy','movenet_thunder','rtmpose_m','yolo11x_pose','sapiens_1b_coco','openpose_coco18'}
EXPECTED_DATASETS={'coco_singleperson_val2017','syrip_test100'}

def main():
    runs=[]
    for manifest_path in (ROOT/'03_model_raw_outputs').glob('*/*/fair_reference/*/run_manifest.json'):
        m=json.loads(manifest_path.read_text(encoding='utf-8'))
        if m.get('profile')!='fair_reference' or m.get('protocol')!=EXPECTED or m.get('profile_id')!='fair_reference_v1' or not m.get('metric_complete'): continue
        required=921 if m.get('dataset_id')=='coco_singleperson_val2017' else 100 if m.get('dataset_id')=='syrip_test100' else None
        if required is None or m.get('counts',{}).get('input')!=required: continue
        runs.append((manifest_path,m))
    if not runs: raise RuntimeError('No complete, comparable fair_reference runs found.')
    latest={}
    for path,m in sorted(runs,key=lambda x:x[1]['run_id']): latest[(m['model_id'],m['dataset_id'])]=(path,m)
    missing=sorted((model,dataset) for model in EXPECTED_MODELS for dataset in EXPECTED_DATASETS if (model,dataset) not in latest)
    if missing: raise RuntimeError(f'Cannot publish the six-model table; complete fair_reference runs are missing: {missing}')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'); out=ROOT/'04_merged_results'/f'merged_{stamp}'; out.mkdir(parents=True)
    rows=[]
    for path,m in latest.values():
        rows.append({'run_id':m['run_id'],'model_id':m['model_id'],'dataset_id':m['dataset_id'],**m['timing'],'handoff_hash':m.get('data_handoff_hash'),'metric_complete':True,'timing_comparability':bool(m.get('timing_comparability')),'fair_ranking_eligible':bool(m.get('fair_ranking_eligible')),'prediction_file':str(path.parent/'predictions.jsonl')})
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (out/'benchmark_table.csv').open('w',newline='',encoding='utf-8-sig') as f: w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    (out/'merge_manifest.json').write_text(json.dumps({'profile_id':'fair_reference_v1','row_count':len(rows),'runs':[r['run_id'] for r in rows]},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(out),'row_count':len(rows),'runs':[r['run_id'] for r in rows]},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
