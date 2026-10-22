from time import perf_counter
import numpy as np
from .base import BaseAdapter
from benchmark_core.contracts import Prediction

class YOLO11xPose(BaseAdapter):
    model_id='yolo11x_pose'
    def load(self,profile):
        from ultralytics import YOLO
        self.model=YOLO(str(getattr(self,'model_path','yolo11x-pose.pt')))
        self.runtime_device='cuda' if __import__('torch').cuda.is_available() else 'cpu'
    def predict(self,row,profile):
        from PIL import Image
        started=perf_counter(); t=perf_counter(); rgb=np.asarray(Image.open(row['image_path']).convert('RGB')); decode=(perf_counter()-t)*1000
        # FP32 is Ultralytics' default; omit the deprecated `half` switch.
        t=perf_counter(); result=self.model.predict(rgb,device=0,imgsz=640,conf=.001,verbose=False)[0]; inf=(perf_counter()-t)*1000; t=perf_counter()
        timing={'decode':decode,'preprocess':0.,'inference':inf,'postprocess':0.,'end_to_end':0.}
        if result.keypoints is None or len(result.keypoints.xy)==0:
            timing['postprocess']=(perf_counter()-t)*1000; timing['end_to_end']=(perf_counter()-started)*1000
            return Prediction(row['sample_id'],row['dataset_id'],self.model_id,'','no_detection',timings_ms=timing,error_code='NO_DETECTION',native={})
        i=int(result.boxes.conf.argmax().item()); pts=result.keypoints.xy[i].cpu().numpy().tolist(); scores=result.keypoints.conf[i].cpu().numpy().tolist(); native={'bbox_xyxy':result.boxes.xyxy[i].cpu().numpy().tolist(),'box_confidence':float(result.boxes.conf[i]),'keypoints':pts,'keypoint_scores':scores}
        timing['postprocess']=(perf_counter()-t)*1000; timing['end_to_end']=(perf_counter()-started)*1000
        return Prediction(row['sample_id'],row['dataset_id'],self.model_id,'','ok',pts,scores,timings_ms=timing,native=native)
    def provenance(self): return {'model_id':self.model_id,'adapter':type(self).__name__,'weight_source':'Ultralytics official yolo11x-pose.pt','runtime_device':getattr(self,'runtime_device','unknown'),'amp':False,'compile':False,'quantization':False,'tensorrt':False,'imgsz':640}
