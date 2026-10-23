"""Run a formal fair_reference dataset/model pair from a model-specific kernel."""
import argparse, json, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark_core.runner import FAIR_REFERENCE, run_dataset

def make_adapter(model_id):
    cache = ROOT / '02_datasets' / 'model_cache'
    if model_id == 'mediapipe_pose_heavy':
        from adapters.mediapipe import MediaPipeHeavy
        a=MediaPipeHeavy(); a.model_path=str(cache/'mediapipe_pose_heavy'/'pose_landmarker_heavy.task'); return a
    if model_id == 'movenet_thunder':
        from adapters.movenet import MoveNetThunder
        return MoveNetThunder(cache/'movenet_thunder'/'movenet_thunder.tflite')
    if model_id == 'rtmpose_m':
        from adapters.rtmpose import RTMPoseM
        a=RTMPoseM(); d=cache/'rtmpose_m'; a.detector_path=d/'yolox_m.onnx'; a.pose_path=d/'rtmpose_m.onnx'; return a
    if model_id == 'yolo11x_pose':
        from adapters.yolo import YOLO11xPose
        a=YOLO11xPose(); a.model_path=str(cache/'yolo11x_pose'/'yolo11x-pose.pt'); return a
    if model_id == 'openpose_coco18':
        from adapters.openpose import OpenPoseCoco18
        d=cache/'openpose_coco18'
        return OpenPoseCoco18(d/'pose_deploy_linevec.prototxt',d/'pose_iter_440000.caffemodel')
    if model_id == 'sapiens_1b_coco':
        from adapters.sapiens import Sapiens1BCoco
        cache_root = Path(os.environ.get('INFANTPOSE_CACHE_ROOT', str(Path.home()/'.local/share/infantpose')))
        model_cache = Path(os.environ.get('INFANTPOSE_SAPIENS_CACHE', str(cache_root/'model_cache'/'sapiens_1b')))
        checkpoint = Path(os.environ.get('INFANTPOSE_SAPIENS_CHECKPOINT', str(model_cache/'sapiens_1b_coco_best_coco_AP_821_torchscript.pt2')))
        demo_root = Path(os.environ.get('INFANTPOSE_SAPIENS_DEMO_ROOT', str(cache_root/'src'/'sapiens')))
        return Sapiens1BCoco(checkpoint, demo_root)
    raise ValueError(f'{model_id} has no validated local Python adapter; see MIGRATION_REPORT.md')

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',required=True,choices=['mediapipe_pose_heavy','movenet_thunder','rtmpose_m','yolo11x_pose','sapiens_1b_coco','openpose_coco18']); p.add_argument('--dataset',required=True,choices=['coco_singleperson_val2017','syrip_test100','coco_unihuman_child_val134']); a=p.parse_args()
    handoff=ROOT/'02_datasets'/'canonical'/a.dataset
    path=run_dataset(make_adapter(a.model),handoff,ROOT/'03_model_raw_outputs','fair_reference',FAIR_REFERENCE)
    manifest=json.loads((path/'run_manifest.json').read_text(encoding='utf-8'))
    status='completed' if manifest.get('metric_complete') else 'incomplete_or_thermally_stopped'
    print(json.dumps({'status':status,'model_id':a.model,'dataset_id':a.dataset,'run_path':str(path),'processed_images':manifest.get('processed_images'),'expected_images':manifest.get('counts',{}).get('input'),'thermal_stop':manifest.get('thermal_stop'),'evaluation_error':manifest.get('evaluation_error')},ensure_ascii=False))
    if status!='completed': raise SystemExit(2)

if __name__=='__main__': main()
