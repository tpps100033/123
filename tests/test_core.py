import contextlib, importlib.util, io, json, tempfile, unittest
from pathlib import Path
from benchmark_core.contracts import Prediction
from benchmark_core.metrics import evaluate_coco_keypoints

class CoreTests(unittest.TestCase):
    def test_prediction_contract(self):
        self.assertEqual(Prediction('s','d','m','r','ok',[[1,2]]*17,[1]*17,timings_ms={'decode':1,'preprocess':1,'inference':1,'postprocess':1,'end_to_end':4}).validate().status,'ok')

    @unittest.skipUnless(importlib.util.find_spec('pycocotools'), 'pycocotools unavailable in this host Python')
    def test_official_cocoeval_perfect_keypoints(self):
        points=[[float(10+i),float(20+i)] for i in range(17)]
        keypoints=[v for point in points for v in (point[0],point[1],2)]
        annotation={'images':[{'id':1,'width':100,'height':100,'file_name':'one.jpg'}],
            'categories':[{'id':1,'name':'person','keypoints':['nose','left_eye','right_eye','left_ear','right_ear','left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle'],'skeleton':[]}],
            'annotations':[{'id':1,'image_id':1,'category_id':1,'keypoints':keypoints,'num_keypoints':17,'bbox':[0,0,100,100],'area':10000,'iscrowd':0,'segmentation':[]}],
            'info':{'description':'fixture'}}
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'annotations.json'; path.write_text(json.dumps(annotation),encoding='utf-8')
            row={'sample_id':'d:1','image_id':'1'}
            pred={'sample_id':'d:1','status':'ok','keypoints_xy':points,'keypoint_scores':[1.0]*17,'native':None}
            with contextlib.redirect_stdout(io.StringIO()):
                summary,_=evaluate_coco_keypoints(path,[row],[pred])
                empty,_=evaluate_coco_keypoints(path,[row],[{'sample_id':'d:1','status':'no_detection'}])
        self.assertEqual(summary['metric_protocol'],"cocodataset/cocoapi.COCOeval(iouType='keypoints')")
        self.assertAlmostEqual(summary['AP'],1.0)
        self.assertAlmostEqual(empty['AP'],0.0)

if __name__=='__main__': unittest.main()
