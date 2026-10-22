from pathlib import Path
from datetime import datetime, timezone
from time import perf_counter
import json, platform, statistics, uuid, logging, os
from .contracts import Prediction, RunManifest
from .data import load_handoff, sha256

FAIR_REFERENCE={'batch_size':1,'warmup_images':10,'precision':'fp32','amp':False,'compile':False,'quantization':False,'tensor_rt':False,'rankable_across_models':True}
from .metrics import evaluate_coco_keypoints, per_image_status

def _write_jsonl(path, rows):
    with open(path,'w',encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')

def _gpu_memory_snapshot():
    """Best-effort framework and device VRAM snapshot; never affects benchmark scores."""
    snapshot={}
    try:
        import torch
        if torch.cuda.is_available():
            snapshot['torch_allocated_mb']=round(torch.cuda.memory_allocated()/1024**2,2)
            snapshot['torch_reserved_mb']=round(torch.cuda.memory_reserved()/1024**2,2)
            snapshot['torch_peak_allocated_mb']=round(torch.cuda.max_memory_allocated()/1024**2,2)
    except Exception as exc:
        snapshot['torch_probe_error']=type(exc).__name__
    try:
        import pynvml
        pynvml.nvmlInit()
        handle=pynvml.nvmlDeviceGetHandleByIndex(int(os.environ.get('CUDA_VISIBLE_DEVICE_INDEX','0')))
        mem=pynvml.nvmlDeviceGetMemoryInfo(handle)
        snapshot.update(device_used_mb=round(mem.used/1024**2,2),device_free_mb=round(mem.free/1024**2,2),temperature_c=pynvml.nvmlDeviceGetTemperature(handle,pynvml.NVML_TEMPERATURE_GPU))
    except Exception as exc:
        snapshot['nvml_probe_error']=type(exc).__name__
    return snapshot

def _gpu_temperature_c():
    try:
        import pynvml
        pynvml.nvmlInit()
        handle=pynvml.nvmlDeviceGetHandleByIndex(int(os.environ.get('CUDA_VISIBLE_DEVICE_INDEX','0')))
        return int(pynvml.nvmlDeviceGetTemperature(handle,pynvml.NVML_TEMPERATURE_GPU))
    except Exception:
        return None

def run_dataset(adapter, handoff_root, output_root, profile, protocol):
    if profile!='fair_reference': raise RuntimeError('正式專案只寫入 fair_reference；原生加速結果不可混入。')
    bad={k:(protocol.get(k),v) for k,v in FAIR_REFERENCE.items() if protocol.get(k)!=v}
    if bad: raise RuntimeError(f'fair_reference 協定被更動：{bad}')
    manifest, rows=load_handoff(handoff_root); run_id=f'{adapter.model_id}_{manifest["dataset_id"]}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:6]}'
    load_started=perf_counter(); load_error=None
    thermal_limit=int(os.environ.get('INFANTPOSE_MAX_GPU_TEMP_C','75')); thermal_stop=None
    try:
        adapter.load(profile)
        adapter.warmup(rows[:10],profile)
        temperature=_gpu_temperature_c()
        if temperature is not None and temperature >= thermal_limit:
            thermal_stop={'temperature_c':temperature,'limit_c':thermal_limit,'processed':0,'phase':'warmup'}
    except Exception as e:
        load_error=e
    load_ms=(perf_counter()-load_started)*1000
    predictions=[]
    for index,row in enumerate(rows):
        if thermal_stop:
            break
        if index % 10 == 0:
            temperature=_gpu_temperature_c()
            if temperature is not None and temperature >= thermal_limit:
                thermal_stop={'temperature_c':temperature,'limit_c':thermal_limit,'processed':len(predictions),'phase':'inference'}
                break
        try:
            if load_error: raise load_error
            p=adapter.predict(row,profile)
        except Exception as e:
            # Keep a failed sample in the denominator and proceed to the next image.
            try:
                import torch
                if torch.cuda.is_available() and isinstance(e,torch.cuda.OutOfMemoryError): torch.cuda.empty_cache()
            except Exception: pass
            p=Prediction(row['sample_id'],manifest['dataset_id'],adapter.model_id,run_id,'model_error',error_code=type(e).__name__,native={'message':str(e)[:1000]},timings_ms={'decode':0.0,'preprocess':0.0,'inference':0.0,'postprocess':0.0,'end_to_end':0.0})
        p.run_id=run_id; p.validate(); predictions.append(p.record())
    timing=[p.get('timings_ms',{}).get('end_to_end') for p in predictions]; timing=[x for x in timing if x is not None]
    evaluation_error=None; metric_rows=[]
    try:
        if thermal_stop:
            raise RuntimeError(f"GPU thermal stop at {thermal_stop['temperature_c']} C (limit {thermal_limit} C); partial runs are not scored.")
        project_root=Path(__file__).resolve().parents[1]
        config=json.loads((project_root/'config/local.json').read_text(encoding='utf-8'))
        annotation_file=project_root/config['datasets'][manifest['dataset_id']]['annotation']
        summary, valid_by_sample=evaluate_coco_keypoints(annotation_file,rows,predictions)
        metric_rows=per_image_status(rows,predictions,valid_by_sample)
    except Exception as e:
        evaluation_error=f'{type(e).__name__}: {e}'
        summary={'metric_protocol':'cocodataset/cocoapi.COCOeval(iouType=keypoints)','evaluation_status':'failed','evaluation_error':evaluation_error}
        metric_rows=[{'sample_id':p['sample_id'],'image_id':next(row.get('image_id',row['sample_id']) for row in rows if row['sample_id']==p['sample_id']),'prediction_status':p['status'],'included_as_detection':False,'error_code':p.get('error_code')} for p in predictions]
    summary['latency_ms']={'n':len(timing),'mean':sum(timing)/len(timing) if timing else None,'median':statistics.median(timing) if timing else None,'p95':sorted(timing)[max(0,int(.95*len(timing))-1)] if timing else None}
    out=Path(output_root)/adapter.model_id/manifest['dataset_id']/profile/run_id; out.mkdir(parents=True)
    env={'platform':platform.platform(),'python':platform.python_version(),'gpu_memory':_gpu_memory_snapshot()}
    try:
        import torch
        env.update(torch=torch.__version__,cuda_available=torch.cuda.is_available(),gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except Exception: pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            env.update(torch=torch.__version__,cuda_available=True,gpu=torch.cuda.get_device_name(0),gpu_capability=list(torch.cuda.get_device_capability(0)))
        else: env.update(torch=torch.__version__,cuda_available=False,gpu=None)
    except Exception: env.update(cuda_available=False,gpu=None)
    counts={'input':len(rows),'ok':len([p for p in predictions if p['status']=='ok']),'no_detection':len([p for p in predictions if p['status']=='no_detection']),'errors':len([p for p in predictions if p['status'] in ('model_error','input_error','mapping_error')])}
    provenance=adapter.provenance()
    expected=921 if manifest['dataset_id']=='coco_singleperson_val2017' else 100 if manifest['dataset_id']=='syrip_test100' else None
    metric_complete=load_error is None and evaluation_error is None and thermal_stop is None and len(predictions)==len(rows) and expected is not None and len(rows)==expected and counts['errors']==0
    runtime_device=str(provenance.get('runtime_device','')).lower()
    timing_comparable=metric_complete and runtime_device in ('gpu','cuda','cuda:0','gpu_delegate')
    rm=RunManifest('InfantPose-Bench-v4','v4.run',run_id,manifest['dataset_id'],adapter.model_id,profile,sha256(Path(handoff_root)/'dataset_manifest.json'),provenance,env,protocol,counts,summary).record()
    rm.update({'data_handoff_hash':manifest['index_sha256'],'weight_source':provenance.get('weight_source'),'metric_complete':metric_complete,'timing_comparability':timing_comparable,'fair_ranking_eligible':timing_comparable,'timing_scope':'decode + preprocessing + inference + postprocessing','warmup':{'images':10,'attempted':min(10,len(rows)),'excluded_from_metrics':True,'errors':getattr(adapter,'warmup_errors',[])},'model_load_ms':load_ms,'model_load_error':None if load_error is None else type(load_error).__name__,'thermal_stop':thermal_stop,'thermal_limit_c':thermal_limit,'processed_images':len(predictions),'profile_id':'fair_reference_v1'})
    rm.update({'evaluation_error':evaluation_error,'metric_protocol':summary.get('metric_protocol'),'metric_protocol_version':summary.get('metric_protocol_version')})
    (out/'run_manifest.json').write_text(json.dumps(rm,indent=2),encoding='utf-8'); _write_jsonl(out/'predictions.jsonl',predictions); _write_jsonl(out/'metrics.jsonl',metric_rows)
    log_root=Path(output_root).parent/'06_logs'; log_root.mkdir(parents=True,exist_ok=True)
    with (log_root/'runs.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps({'run_id':run_id,'model_id':adapter.model_id,'dataset_id':manifest['dataset_id'],'path':str(out),'counts':rm['counts'],'timing':summary,'gpu_memory':env.get('gpu_memory'),'model_load_error':rm['model_load_error']},ensure_ascii=False)+'\n')
    return out
