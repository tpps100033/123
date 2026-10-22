import csv, json, tempfile, unittest
from pathlib import Path
from benchmark_core.data import sha256
from benchmark_core.runner import FAIR_REFERENCE, run_dataset
from adapters.base import BaseAdapter
from benchmark_core.contracts import Prediction

class BrokenAdapter(BaseAdapter):
    model_id='test_adapter'
    def predict(self,row,profile): raise RuntimeError('simulated per-image failure')
    def provenance(self): return {'model_id':self.model_id,'runtime_device':'cpu'}

class RunnerTests(unittest.TestCase):
    def test_per_image_failure_is_recorded_without_rank_eligibility(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); handoff=root/'handoff'; handoff.mkdir()
            index=handoff/'sample_index.csv'
            image=root/'sample.jpg';image.write_bytes(b'fixture')
            with index.open('w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=['sample_id','dataset_id','image_path','gt_xy','visibility','bbox_xywh']);w.writeheader();w.writerow({'sample_id':'x','dataset_id':'tiny','image_path':str(image),'gt_xy':json.dumps([[0,0]]*17),'visibility':json.dumps([2]*17),'bbox_xywh':json.dumps([0,0,10,10])})
            (handoff/'dataset_manifest.json').write_text(json.dumps({'dataset_id':'tiny','index':'sample_index.csv','index_sha256':sha256(index)}),encoding='utf-8')
            path=run_dataset(BrokenAdapter(),handoff,root/'03_model_raw_outputs','fair_reference',FAIR_REFERENCE)
            manifest=json.loads((path/'run_manifest.json').read_text(encoding='utf-8'))
            prediction=json.loads((path/'predictions.jsonl').read_text(encoding='utf-8'))
            self.assertEqual(prediction['status'],'model_error')
            self.assertFalse(manifest['metric_complete'])
            self.assertFalse(manifest['timing_comparability'])
            self.assertEqual(manifest['counts']['errors'],1)
            self.assertTrue((root/'06_logs'/'runs.jsonl').is_file())

    def test_batch_interface_returns_each_single_image_result(self):
        class ValidAdapter(BaseAdapter):
            def predict(self,row,profile): return Prediction(row['sample_id'],'d','m','r','no_detection',timings_ms={'decode':0,'preprocess':0,'inference':0,'postprocess':0,'end_to_end':0})
        result=ValidAdapter().predict_batch([{'sample_id':'a'},{'sample_id':'b'}],'fair_reference')
        self.assertEqual([x.sample_id for x in result],['a','b'])

if __name__=='__main__': unittest.main()
