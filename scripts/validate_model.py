"""Real inference smoke check: one image and a two-image sequential batch."""
import argparse, json, sys, traceback
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from benchmark_core.data import load_handoff
from scripts.run_research import make_adapter

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',required=True,choices=['mediapipe_pose_heavy','movenet_thunder','rtmpose_m','yolo11x_pose','openpose_coco18']); p.add_argument('--dataset',default='coco_singleperson_val2017',choices=['coco_singleperson_val2017','syrip_test100']); a=p.parse_args()
    result={'model_id':a.model,'dataset_id':a.dataset,'started_utc':datetime.now(timezone.utc).isoformat(),'status':'failed','single_image':None,'batch_two':None,'gpu_used':False}
    try:
        adapter=make_adapter(a.model); handoff,rows=load_handoff(ROOT/'02_datasets'/'canonical'/a.dataset)
        if len(rows)<2: raise RuntimeError('Smoke validation needs at least two rows in the verified handoff.')
        adapter.load('fair_reference'); result['model_provenance']=adapter.provenance()
        one=adapter.predict(rows[0],'fair_reference').validate(); two=adapter.predict_batch(rows[:2],'fair_reference')
        if len(two)!=2: raise RuntimeError(f'batch API returned {len(two)} outputs for 2 images')
        for prediction in two: prediction.validate()
        result['single_image']={'status':one.status,'keypoint_count':len(one.keypoints_xy or []),'timings_ms':one.timings_ms}
        result['batch_two']={'count':len(two),'statuses':[x.status for x in two],'keypoint_counts':[len(x.keypoints_xy or []) for x in two]}
        device=str(result['model_provenance'].get('runtime_device','')).lower(); result['gpu_used']=device in ('gpu','cuda','cuda:0','gpu_delegate')
        result['gpu_evidence']={'runtime_device':result['model_provenance'].get('runtime_device'),'fallback_reason':result['model_provenance'].get('gpu_fallback_reason')}
        first=two[0]
        consistent=first.status==one.status
        if consistent and one.keypoints_xy and first.keypoints_xy:
            consistent=len(one.keypoints_xy)==len(first.keypoints_xy) and all(len(x)==len(y)==2 and max(abs(float(x[i])-float(y[i])) for i in range(2))<=0.01 for x,y in zip(one.keypoints_xy,first.keypoints_xy))
        result['batch_single_consistent']=consistent
        valid_statuses={'ok','no_detection'}
        outputs_valid=one.status in valid_statuses and all(x.status in valid_statuses for x in two) and result['batch_two']['keypoint_counts']==[0 if x.status=='no_detection' else 17 for x in two]
        try:
            import torch
            result['torch_cuda_available']=torch.cuda.is_available()
            if torch.cuda.is_available(): result['torch_device']=torch.cuda.get_device_name(0)
        except Exception: pass
        result['status']='inference_pass' if outputs_valid and consistent and result['gpu_used'] else 'inference_pass_cpu_only' if outputs_valid and consistent else 'format_failed'
    except Exception as exc:
        result.update(status='blocked' if 'no local adapter/runtime' in str(exc) else 'failed',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(limit=5))
    finally:
        if 'adapter' in locals() and hasattr(adapter,'close'):
            try: adapter.close()
            except Exception as exc: result['close_error']=f'{type(exc).__name__}: {exc}'
    log=ROOT/'06_logs'; log.mkdir(exist_ok=True); out=log/f"model_validation_{a.model}.json"
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(result,ensure_ascii=False,indent=2))
    if result['status'] not in {'inference_pass','inference_pass_cpu_only'}: raise SystemExit(1)

if __name__=='__main__': main()
