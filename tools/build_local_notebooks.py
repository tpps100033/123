"""Create portable local notebook entry points; original Colab notebooks stay intact."""
import json
import uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'notebooks'/'local'; OUT.mkdir(parents=True,exist_ok=True)

def notebook(filename,title,markdown,cells,kernel='infantpose-rtmpose',display_name='InfantPose (rtmpose)'):
    content=[{'cell_type':'markdown','id':uuid.uuid4().hex[:8],'metadata':{},'source':(f'# {title}\n\n{markdown}\n').splitlines(True)}]
    content += [{'cell_type':'code','id':uuid.uuid4().hex[:8],'metadata':{},'execution_count':None,'outputs':[],'source':s.splitlines(True)} for s in cells]
    doc={'cells':content,'metadata':{'kernelspec':{'display_name':display_name,'language':'python','name':kernel}},'nbformat':4,'nbformat_minor':5}
    (OUT/filename).write_text(json.dumps(doc,ensure_ascii=False,indent=2),encoding='utf-8')

boot="""from pathlib import Path
ROOT = next((path for path in (Path.cwd(), *Path.cwd().parents) if (path / 'config' / 'local.json').is_file()), None)
assert ROOT is not None, 'Open this notebook within the InfantPose project tree.'
print('Project:', ROOT)
"""
notebook('00_Prepare_TwoDomain_Handoffs_local.ipynb','00｜本地資料與模型快取準備','此入口使用專案內的相對路徑，原始 Colab notebook 保存在 `../original/`。先將原始資料集與標註放到 `02_datasets/source_archives/`，再執行下方兩個步驟。評估資料數量與 COCO-17 對應不符時會停止。',[boot,"""import subprocess, sys
subprocess.run([sys.executable, str(ROOT/'scripts'/'prepare_datasets.py')], check=True)
subprocess.run([sys.executable, str(ROOT/'scripts'/'prepare_model_assets.py')], check=True)
"""], 'infantpose-rtmpose', 'InfantPose (rtmpose)')
models=[('01_MediaPipe_Heavy','mediapipe_pose_heavy','MediaPipe Heavy','infantpose-mediapipe'),('02_MoveNet_Thunder','movenet_thunder','MoveNet Thunder','infantpose-movenet'),('03_RTMPose_M','rtmpose_m','RTMPose-M','infantpose-rtmpose'),('04_YOLO11x_Pose','yolo11x_pose','YOLO11x-Pose','infantpose-yolo')]
for stem,model,title,kernel in models:
    cells=[boot,"""import subprocess, sys
MODEL = '"""+model+"""'
DATASETS = ('coco_singleperson_val2017', 'syrip_test100')
for dataset in DATASETS:
    subprocess.run([sys.executable, str(ROOT/'scripts'/'run_research.py'), '--model', MODEL, '--dataset', dataset], check=True)
"""]
    notebook(stem+'_FairReference_local.ipynb',title+'｜本地 fair_reference',f'請選擇 **{kernel}** kernel。正式協定固定 batch=1、10 張 warm-up、FP32；指標由官方 `pycocotools.COCOeval(iouType="keypoints")` 計算。COCO 結果是 921 張單人子集上的 AP，不代表完整 val2017 排行榜成績；SyRIP 標示為自訂資料域的 COCO OKS 評估。執行前先完成 `00`，並確認模型快取與兩個已驗證資料 handoff 均存在。',[*cells],kernel,'InfantPose ('+kernel.rsplit('-',1)[-1]+')')
notebook('05_Sapiens_1B_local.ipynb','05｜Sapiens 1B CUDA 評估','此模型在原始研究 Drive 中屬於六模型基準；PRD v4 將它排除於原四模型 scope，但本地擴充流程保留並實際評估。請先從原研究 Drive 快取取得 checkpoint，並在 WSL cache 安裝 Sapiens Lite 官方 demo。使用獨立 `infantpose-sapiens` kernel、FP32、batch=1 與相同官方 COCOeval。',[boot,"""import subprocess, sys
MODEL = 'sapiens_1b_coco'
CHECKPOINT = Path.home() / '.local/share/infantpose/model_cache/sapiens_1b/sapiens_1b_coco_best_coco_AP_821_torchscript.pt2'
assert CHECKPOINT.is_file() and CHECKPOINT.stat().st_size == 4676154239, f'Sapiens checkpoint unavailable or incomplete: {CHECKPOINT}'
subprocess.run([sys.executable, str(ROOT/'scripts'/'validate_model.py'), '--model', MODEL, '--dataset', 'coco_singleperson_val2017'], check=True)
for dataset in ('coco_singleperson_val2017', 'syrip_test100'):
    subprocess.run([sys.executable, str(ROOT/'scripts'/'run_research.py'), '--model', MODEL, '--dataset', dataset], check=True)
"""], 'infantpose-sapiens', 'InfantPose (sapiens)')
notebook('06_OpenPose_COCO18_local.ipynb','06｜OpenPose COCO-18 CUDA 評估','使用 WSL2 中本地編譯的 OpenPose 1.7.0 CUDA/Caffe runtime。固定網路輸入為 480×368 以限制 RTX 5060 顯存；先執行單張與雙張推論驗證，再跑完整 COCO 與 SyRIP 官方 COCOeval。',[boot,"""import subprocess, sys
subprocess.run([sys.executable, str(ROOT/'scripts'/'validate_model.py'), '--model', 'openpose_coco18', '--dataset', 'coco_singleperson_val2017'], check=True)
for dataset in ('coco_singleperson_val2017', 'syrip_test100'):
    subprocess.run([sys.executable, str(ROOT/'scripts'/'run_research.py'), '--model', 'openpose_coco18', '--dataset', dataset], check=True)
"""], 'infantpose-openpose', 'InfantPose (openpose)')
notebook('90_Merge_FairReference_local.ipynb','90｜合併公平基準結果','只接受完整 921/100 張資料集的 `fair_reference_v1` runs。歷史匯入資料不會加入跨模型 timing 排名。',[boot,"""import subprocess, sys
subprocess.run([sys.executable, str(ROOT/'scripts'/'merge_results.py')], check=True)
"""], 'infantpose-rtmpose', 'InfantPose (rtmpose)')
notebook('91_Analyse_TwoDomain_FairReference_local.ipynb','91｜兩域結果分析','只整理 COCO 子集 AP 與 SyRIP COCO-OKS 評估器的實際成對輸出；缺失分數不補造。',[boot,"""import subprocess, sys
subprocess.run([sys.executable, str(ROOT/'scripts'/'analyse_results.py')], check=True)
"""], 'infantpose-rtmpose', 'InfantPose (rtmpose)')
print(f'Wrote local notebook entry points to {OUT}')
