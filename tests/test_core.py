import csv,json,tempfile,unittest
from pathlib import Path
from benchmark_core.contracts import Prediction
from benchmark_core.metrics import score
class CoreTests(unittest.TestCase):
 def test_prediction_contract(self): self.assertEqual(Prediction('s','d','m','r','ok',[[1,2]]*17,[1]*17,timings_ms={'decode':1,'preprocess':1,'inference':1,'postprocess':1,'end_to_end':4}).validate().status,'ok')
 def test_failed_is_zero(self):
  row={'sample_id':'x','gt_xy':json.dumps([[0,0]]*17),'visibility':json.dumps([2]*17),'bbox_xywh':json.dumps([0,0,10,10])}; self.assertEqual(score(row,Prediction('x','d','m','r','no_detection',timings_ms={'decode':0,'preprocess':0,'inference':0,'postprocess':0,'end_to_end':0}))['oks_e2e'],0.)
if __name__=='__main__': unittest.main()
