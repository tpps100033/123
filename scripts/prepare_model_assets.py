"""Download model files once into the project cache; never overwrite valid files."""
import hashlib, json, shutil, urllib.request, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; CACHE=ROOT/'02_datasets'/'model_cache'

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()

def download(url,path,validate=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.is_file() and path.stat().st_size and (validate is None or validate(path)):
        return {'state':'reused','path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,'sha256':digest(path)}
    part=path.with_suffix(path.suffix+'.part'); part.unlink(missing_ok=True)
    try:
        urllib.request.urlretrieve(url,part)
        if not part.stat().st_size or (validate and not validate(part)): raise RuntimeError(f'Asset validation failed: {path.name}')
        part.replace(path)
    finally: part.unlink(missing_ok=True)
    return {'state':'downloaded','path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,'sha256':digest(path),'url':url}

def zip_has_onnx(path):
    try:
        with zipfile.ZipFile(path) as z: return z.testzip() is None and len([i for i in z.infolist() if i.filename.endswith('.onnx') and not i.is_dir()])==1
    except Exception: return False

def extract_onnx(archive,target):
    if target.is_file() and target.stat().st_size>1024*1024: return {'state':'reused','path':str(target.relative_to(ROOT)),'bytes':target.stat().st_size,'sha256':digest(target)}
    with zipfile.ZipFile(archive) as z:
        items=[i for i in z.infolist() if i.filename.endswith('.onnx') and not i.is_dir()]
        if len(items)!=1: raise RuntimeError(f'Expected one ONNX file in {archive.name}; found {len(items)}')
        target.parent.mkdir(parents=True,exist_ok=True); part=target.with_suffix('.part')
        with z.open(items[0]) as src,part.open('wb') as dst: shutil.copyfileobj(src,dst)
        part.replace(target)
    return {'state':'extracted','path':str(target.relative_to(ROOT)),'bytes':target.stat().st_size,'sha256':digest(target)}

def main():
    assets={}
    assets['mediapipe_pose_heavy']=download('https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task',CACHE/'mediapipe_pose_heavy'/'pose_landmarker_heavy.task')
    assets['movenet_thunder']=download('https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/4?lite-format=tflite',CACHE/'movenet_thunder'/'movenet_thunder.tflite')
    assets['yolo11x_pose']=download('https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-pose.pt',CACHE/'yolo11x_pose'/'yolo11x-pose.pt')
    d=CACHE/'rtmpose_m'; d.mkdir(parents=True,exist_ok=True)
    pose_url='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip'
    det_url='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_m_8xb8-300e_humanart-c2c7a14a.zip'
    pose_onnx,det_onnx=d/'rtmpose_m.onnx',d/'yolox_m.onnx'
    if pose_onnx.is_file() and pose_onnx.stat().st_size>1024*1024:
        assets['rtmpose_pose_onnx']={'state':'reused','path':str(pose_onnx.relative_to(ROOT)),'bytes':pose_onnx.stat().st_size,'sha256':digest(pose_onnx)}
    else:
        assets['rtmpose_pose_zip']=download(pose_url,d/'rtmpose_m_sdk.zip',zip_has_onnx)
        assets['rtmpose_pose_onnx']=extract_onnx(d/'rtmpose_m_sdk.zip',pose_onnx)
    if det_onnx.is_file() and det_onnx.stat().st_size>1024*1024:
        assets['rtmpose_detector_onnx']={'state':'reused','path':str(det_onnx.relative_to(ROOT)),'bytes':det_onnx.stat().st_size,'sha256':digest(det_onnx)}
    else:
        assets['rtmpose_detector_zip']=download(det_url,d/'yolox_m_detector_sdk.zip',zip_has_onnx)
        assets['rtmpose_detector_onnx']=extract_onnx(d/'yolox_m_detector_sdk.zip',det_onnx)
    out=CACHE/'cache_status.json'; out.write_text(json.dumps({'schema':'local-model-cache-v1','assets':assets},ensure_ascii=False,indent=2),encoding='utf-8')
    print(out.read_text(encoding='utf-8'))

if __name__=='__main__': main()
