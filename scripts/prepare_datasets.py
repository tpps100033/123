"""Create verified local handoffs from the preserved dataset archives."""
import argparse, json, zipfile, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import json
from benchmark_core.data import build_coco_handoff

def safe_extract(archive, destination):
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / '.archive.sha256'
    from benchmark_core.data import sha256
    digest = sha256(archive)
    if marker.is_file() and marker.read_text(encoding='utf-8').strip() == digest:
        return 'reused'
    root = destination.resolve()
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            path = (destination / info.filename).resolve()
            if not path.is_relative_to(root):
                raise RuntimeError(f'Unsafe path in archive: {info.filename}')
        zf.extractall(destination)
    marker.write_text(digest, encoding='utf-8')
    return 'extracted'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['all', 'coco_singleperson_val2017', 'syrip_test100'], default='all')
    args = parser.parse_args()
    config = json.loads((ROOT / 'config/local.json').read_text(encoding='utf-8'))
    for dataset_id, spec in config['datasets'].items():
        if args.dataset != 'all' and args.dataset != dataset_id:
            continue
        canonical = ROOT / '02_datasets' / 'canonical' / dataset_id
        manifest_path = canonical / 'dataset_manifest.json'
        if manifest_path.is_file():
            manifest, rows = __import__('benchmark_core.data', fromlist=['load_handoff']).load_handoff(canonical)
            expected = spec['expected_samples']
            if len(rows) == expected and manifest.get('mapping_version') == spec['mapping']:
                print(json.dumps({'dataset_id':dataset_id,'archive_status':'reused_verified_handoff','sample_count':len(rows),'index_sha256':manifest['index_sha256']},ensure_ascii=False))
                continue
            raise RuntimeError(f'{dataset_id} has a handoff but count/mapping does not match frozen protocol; preserve it for review instead of rebuilding over it.')
        archive, annotation = ROOT / spec['raw'], ROOT / spec['annotation']
        if not archive.is_file():
            raise FileNotFoundError(f'{dataset_id} source archive missing: {archive}')
        if not annotation.is_file():
            raise FileNotFoundError(f'{dataset_id} annotation JSON missing: {annotation}')
        raw_root = ROOT / '02_datasets' / 'canonical' / dataset_id / 'raw'
        status = safe_extract(archive, raw_root)
        manifest = build_coco_handoff(dataset_id, raw_root, annotation, raw_root.parent, spec['expected_samples'])
        manifest['mapping_version'] = spec['mapping']
        (raw_root.parent / 'dataset_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'dataset_id': dataset_id, 'archive_status': status, 'manifest': manifest}, ensure_ascii=False))

if __name__ == '__main__':
    main()
