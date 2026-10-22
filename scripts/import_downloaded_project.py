"""Import only the frozen sample images, manifests, annotations and model files from Drive export ZIPs."""
import argparse, csv, hashlib, json, stat, zipfile
from pathlib import Path, PurePosixPath
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from benchmark_core.data import sha256

DATASETS=('coco_singleperson_val2017','syrip_test100')
MODEL_FILES=(
    'mediapipe_pose_heavy/pose_landmarker_heavy.task',
    'movenet_thunder/movenet_thunder.tflite',
    'rtmpose_m/rtmpose_m.onnx','rtmpose_m/yolox_m.onnx',
    'yolo11x_pose/yolo11x-pose.pt',
    'openpose_coco18/pose_deploy_linevec.prototxt','openpose_coco18/pose_iter_440000.caffemodel',
)
SOURCE_ARCHIVES=('COCO_SinglePerson_val2017_921.zip','SyRIP.zip')

def safe_name(name):
    p=PurePosixPath(name)
    return not p.is_absolute() and '..' not in p.parts and '\\' not in name and not name.startswith('/')

def main():
    p=argparse.ArgumentParser();p.add_argument('--source-dir',required=True);args=p.parse_args()
    archives=sorted(Path(args.source_dir).glob('*.zip'))
    if not archives: raise FileNotFoundError(f'No source ZIPs in {args.source_dir}')
    selected={}; indices={}; members={}; historical_map={}; canonical_prefix='姿態判定/02_datasets/canonical/'
    for archive in archives:
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                if not safe_name(info.filename): raise RuntimeError(f'Unsafe archive path: {archive.name}:{info.filename}')
                if info.is_dir(): continue
                n=info.filename
                members[n]=(archive,n)
                for dataset_id in DATASETS:
                    if n==f'{canonical_prefix}{dataset_id}/sample_index.csv':
                        if dataset_id in indices: raise RuntimeError(f'Duplicate index for {dataset_id}')
                        indices[dataset_id]=(archive,n)
                if n.startswith('姿態判定/02_datasets/person_keypoints_') and n.endswith('.json'):
                    selected[n]=(archive,n)
                if n.startswith('姿態判定/02_datasets/source_archives/') and Path(n).name in SOURCE_ARCHIVES:
                    selected[n]=(archive,n)
                for source_root,target_root in (('03_model_raw_outputs/','03_model_raw_outputs/legacy_colab_archive/'),('04_merged_results/','04_merged_results/legacy_colab_archive/'),('05_analysis_results/','05_analysis_results/legacy_colab_archive/')):
                    marker='姿態判定/'+source_root
                    if n.startswith(marker):
                        bucket=source_root[:2]
                        digest=hashlib.sha256(n.encode('utf-8')).hexdigest()[:16]
                        target=f'姿態判定/{target_root}{bucket}/{digest}_{Path(n).name}'
                        selected[target]=(archive,n)
                        historical_map[target]=n
                if n.startswith('姿態判定/02_datasets/model_cache/'):
                    rel=n.split('/02_datasets/model_cache/',1)[1]
                    if rel in MODEL_FILES: selected[n]=(archive,n)
    if set(indices)!=set(DATASETS): raise RuntimeError(f'Missing source sample indexes: {set(DATASETS)-set(indices)}')
    rows_by_dataset={}
    for dataset_id,(archive,n) in indices.items():
        with zipfile.ZipFile(archive) as zf: rows=list(csv.DictReader(zf.read(n).decode('utf-8-sig').splitlines()))
        expected=921 if dataset_id==DATASETS[0] else 100
        if len(rows)!=expected: raise RuntimeError(f'{dataset_id}: source has {len(rows)} rows, expected {expected}')
        rows_by_dataset[dataset_id]=rows
        prefix=f'{canonical_prefix}{dataset_id}/'
        for row in rows:
            old=row['image_path'].replace('\\','/')
            marker=f'/canonical/{dataset_id}/'
            if marker not in old: raise RuntimeError(f"Unexpected source image path for {row['sample_id']}: {old}")
            rel=old.split(marker,1)[1]
            image_name=prefix+rel
            # Resolve the member in the same collection below, then select its source archive.
            selected[image_name]=None
        selected[n]=(archive,n)
        manifest_name=f'{canonical_prefix}{dataset_id}/dataset_manifest.json'
        if manifest_name not in members: raise FileNotFoundError(f'Missing frozen manifest for {dataset_id}')
        selected[manifest_name]=members[manifest_name]
    # Map every referenced image to exactly one archive member before writing anything.
    for dataset_id,rows in rows_by_dataset.items():
        prefix=f'{canonical_prefix}{dataset_id}/'
        for row in rows:
            rel=row['image_path'].replace('\\','/').split(f'/canonical/{dataset_id}/',1)[1]
            target_name=prefix+rel
            if target_name not in members: raise FileNotFoundError(f'{row["sample_id"]}: missing image member {target_name}')
            selected[target_name]=members[target_name]
    missing_assets=[]
    for rel in MODEL_FILES:
        name='姿態判定/02_datasets/model_cache/'+rel
        if name not in selected: missing_assets.append(rel)
    if missing_assets: raise FileNotFoundError('Missing cached model files: '+', '.join(missing_assets))
    found_archives={Path(name).name for name in selected if name.startswith('姿態判定/02_datasets/source_archives/')}
    if set(SOURCE_ARCHIVES)-found_archives: raise FileNotFoundError('Missing source dataset archives: '+', '.join(set(SOURCE_ARCHIVES)-found_archives))
    print(f'Extracting {len(selected)} selected project files from {len(archives)} source archives.')
    for target_name,(archive,member) in selected.items():
        relative=PurePosixPath(target_name).relative_to('姿態判定')
        target=ROOT.joinpath(*relative.parts);target.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(archive) as zf:
            info=zf.getinfo(member)
            if stat.S_ISLNK(info.external_attr >> 16): raise RuntimeError(f'Symlink not accepted: {member}')
            with zf.open(info) as src,target.with_suffix(target.suffix+'.part').open('wb') as dst:
                while block:=src.read(8*1024*1024): dst.write(block)
        target.with_suffix(target.suffix+'.part').replace(target)
    log_dir=ROOT/'06_logs';log_dir.mkdir(parents=True,exist_ok=True)
    (log_dir/'historical_colab_index.json').write_text(json.dumps(historical_map,ensure_ascii=False,indent=2),encoding='utf-8')
    for dataset_id,rows in rows_by_dataset.items():
        handoff=ROOT/'02_datasets'/'canonical'/dataset_id
        original=handoff/'sample_index.csv'; (handoff/'sample_index_colab.csv').write_bytes(original.read_bytes())
        for row in rows:
            old=row['image_path'].replace('\\','/'); row['image_path']=old.split(f'/canonical/{dataset_id}/',1)[1]
        with original.open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        manifest_path=handoff/'dataset_manifest.json'; manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
        manifest['source_index_sha256']=manifest.get('index_sha256');manifest['index_sha256']=sha256(original);manifest['path_mode']='relative_to_dataset_handoff'
        manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
        for row in rows:
            image=handoff/row['image_path']
            if not image.is_file(): raise FileNotFoundError(image)
        print(json.dumps({'dataset_id':dataset_id,'sample_count':len(rows),'index_sha256':manifest['index_sha256'],'source_index_sha256':manifest['source_index_sha256'],'image_paths':'verified'},ensure_ascii=False))
    cache=ROOT/'02_datasets'/'model_cache'; assets={}
    for rel in MODEL_FILES:
        f=cache/rel; assets[rel]={'bytes':f.stat().st_size,'sha256':sha256(f)}
    (cache/'cache_status_local_import.json').write_text(json.dumps({'source':'user-provided Google Drive export ZIPs','assets':assets},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':'imported','assets':assets},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
