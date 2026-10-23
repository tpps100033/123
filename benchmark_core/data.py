from pathlib import Path
import csv, hashlib, json

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def _images(raw): return {p.name:p for p in Path(raw).rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp'}}

def build_coco_handoff(dataset_id, raw_root, annotation_json, canonical_root, expected_samples):
    """Build the only acceptable handoff: one image + one COCO-17 annotation per row."""
    root=Path(canonical_root).resolve()
    source=json.loads(Path(annotation_json).read_text(encoding='utf-8'))
    images={str(x['id']):x for x in source.get('images',[])}; found=_images(raw_root); by_image={}
    for a in source.get('annotations',[]):
        if len(a.get('keypoints',[]))==51 and a.get('num_keypoints',0)>0: by_image.setdefault(str(a['image_id']),[]).append(a)
    rows=[]
    for image_id, image in images.items():
        candidates=by_image.get(image_id,[]); path=found.get(Path(image.get('file_name','')).name)
        if not candidates or not path: continue
        a=max(candidates,key=lambda x:x.get('area',0)); k=a['keypoints']
        portable_path=path.resolve().relative_to(root).as_posix()
        rows.append({'sample_id':f'{dataset_id}:{image_id}','dataset_id':dataset_id,'image_id':image_id,'image_path':portable_path,'width':image.get('width'),'height':image.get('height'),'bbox_xywh':json.dumps(a['bbox']),'gt_xy':json.dumps([[k[i],k[i+1]] for i in range(0,51,3)]),'visibility':json.dumps([k[i] for i in range(2,51,3)])})
    if len(rows) != expected_samples: raise RuntimeError(f'{dataset_id}: expected exactly {expected_samples} verified image/COCO17 pairs, found {len(rows)}')
    root.mkdir(parents=True,exist_ok=True); index=root/'sample_index.csv'
    with index.open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    manifest={'schema_version':'v4.dataset-handoff','dataset_id':dataset_id,'sample_count':len(rows),'index':'sample_index.csv','index_sha256':sha256(index),'annotation_contract':'COCO-17','source_annotation_sha256':sha256(annotation_json)}
    (root/'dataset_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8'); return manifest

def load_handoff(root):
    root=Path(root); m=json.loads((root/'dataset_manifest.json').read_text())
    if sha256(root/m['index']) != m['index_sha256']: raise RuntimeError('handoff index hash mismatch')
    with (root/m['index']).open(encoding='utf-8') as f: rows=list(csv.DictReader(f))
    for row in rows:
        image=Path(row['image_path'])
        if not image.is_file() and not image.is_absolute(): image=root/image
        if not image.is_file(): raise FileNotFoundError(f"handoff image missing for {row['sample_id']}: {image}")
        row['image_path']=str(image.resolve())
    return m,rows
