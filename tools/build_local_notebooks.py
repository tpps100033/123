"""Create portable local notebook entry points; original Colab notebooks stay intact."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'notebooks'/'local'; OUT.mkdir(parents=True,exist_ok=True)

def notebook(filename,title,markdown,cells):
    content=[{'cell_type':'markdown','metadata':{},'source':(f'# {title}\n\n{markdown}\n').splitlines(True)}]
    content += [{'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':s.splitlines(True)} for s in cells]
    doc={'cells':content,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'nbformat':4,'nbformat_minor':5}
    (OUT/filename).write_text(json.dumps(doc,ensure_ascii=False,indent=2),encoding='utf-8')

boot="""from pathlib import Path
ROOT = Path.cwd()
assert (ROOT / 'config' / 'local.json').is_file(), 'Run JupyterLab from the InfantPose project root.'
print('Project:', ROOT)
"""
notebook('00_Prepare_TwoDomain_Handoffs_local.ipynb','00｜本地資料與模型快取準備','此入口使用專案內的相對路徑，原始 Colab notebook 保存在 `../original/`。先將原始資料集與標註放到 `02_datasets/source_archives/`，再執行下方兩個步驟。評估資料數量與 COCO-17 對應不符時會停止。',[boot,"""import subprocess, sys
subprocess.run([sys.executable, str(ROOT/'scripts'/'prepare_datasets.py')], check=True)
subprocess.run([sys.executable, str(ROOT/'scripts'/'prepare_model_assets.py')], check=True)
"""])
models=[('01_MediaPipe_Heavy','mediapipe_pose_heavy','MediaPipe Heavy','infantpose-mediapipe'),('02_MoveNet_Thunder','movenet_thunder','MoveNet Thunder','infantpose-movenet'),('03_RTMPose_M','rtmpose_m','RTMPose-M','infantpose-rtmpose'),('04_YOLO11x_Pose','yolo11x_pose','YOLO11x-Pose','infantpose-yolo')]
for stem,model,title,kernel in models:
    cells=[boot,"""import subprocess, sys
MODEL = '"""+model+"""'
DATASETS = ('coco_singleperson_val2017', 'syrip_test100')
for dataset in DATASETS:
    subprocess.run([sys.executable, str(ROOT/'scripts'/'run_research.py'), '--model', MODEL, '--dataset', dataset], check=True)
"""]
    notebook(stem+'_FairReference_local.ipynb',title+'｜本地 fair_reference',f'請選擇 **{kernel}** kernel。正式協定固定 batch=1、10 張 warm-up、FP32；不修改評估定義。執行前先完成 `00`，並確認模型快取與兩個已驗證資料 handoff 均存在。',[*cells])
notebook('05_Sapiens_1B_local.ipynb','05｜Sapiens 1B（非 v4 正式範圍）','原始 Colab Notebook 保留在 `../original/`。PRD v4 明確將 Sapiens 排除在正式模型範圍外；本地不載入 gated 權重、不產生比較分數，也不把 import 當成通過測試。',[boot,"print({'model_id':'sapiens_1b_coco','status':'not_in_v4_formal_scope','source_notebook':'../original/05_Sapiens_1B_FairReference_v4.ipynb'})"])
notebook('06_OpenPose_COCO18_local.ipynb','06｜OpenPose COCO-18 相容性驗證','原始 Notebook 使用 Colab/Linux OpenPose v1.7.0 執行檔。RTX 5060 / WSL2 的本地 runtime 尚未建置；此入口會寫出 blocked 狀態，不會偽造推論通過。',[boot,"""import subprocess, sys
subprocess.run([sys.executable, str(ROOT/'scripts'/'validate_model.py'), '--model', 'openpose_coco18'], check=False)
"""])
notebook('90_Merge_FairReference_local.ipynb','90｜合併公平基準結果','只接受完整 921/100 張資料集的 `fair_reference_v1` runs。歷史匯入資料不會加入跨模型 timing 排名。',[boot,"""import subprocess, sys
subprocess.run([sys.executable, str(ROOT/'scripts'/'merge_results.py')], check=True)
"""])
notebook('91_Analyse_TwoDomain_FairReference_local.ipynb','91｜兩域結果分析','只整理 COCO val2017 與 SyRIP Test100 的實際成對輸出；缺失分數不補造。',[boot,"""import subprocess, sys
subprocess.run([sys.executable, str(ROOT/'scripts'/'analyse_results.py')], check=True)
"""])
print(f'Wrote local notebook entry points to {OUT}')
