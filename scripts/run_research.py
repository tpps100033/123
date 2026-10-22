"""Run a formal fair_reference dataset/model pair from a model-specific kernel."""
import argparse, json, sys
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
    raise ValueError(f'{model_id} has no validated local Python adapter; see MIGRATION_REPORT.md')

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',required=True,choices=['mediapipe_pose_heavy','movenet_thunder','rtmpose_m','yolo11x_pose']); p.add_argument('--dataset',required=True,choices=['coco_singleperson_val2017','syrip_test100']); a=p.parse_args()
    handoff=ROOT/'02_datasets'/'canonical'/a.dataset
    path=run_dataset(make_adapter(a.model),handoff,ROOT/'03_model_raw_outputs','fair_reference',FAIR_REFERENCE)
    print(json.dumps({'status':'completed','model_id':a.model,'dataset_id':a.dataset,'run_path':str(path)},ensure_ascii=False))

if __name__=='__main__': main()
