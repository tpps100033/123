from time import perf_counter
import numpy as np
from .base import BaseAdapter
from benchmark_core.contracts import Prediction

class RTMPoseM(BaseAdapter):
    model_id='rtmpose_m'
    def load(self,profile):
        from rtmlib import YOLOX, RTMPose
        import onnxruntime as ort
        providers=ort.get_available_providers()
        self.runtime_device='cuda' if 'CUDAExecutionProvider' in providers else 'cpu'
        self.detector=YOLOX(str(self.detector_path),model_input_size=(640,640),backend='onnxruntime',device='cuda:0' if self.runtime_device=='cuda' else 'cpu')
        self.pose=RTMPose(str(self.pose_path),model_input_size=(192,256),backend='onnxruntime',device='cuda:0' if self.runtime_device=='cuda' else 'cpu')
    def predict(self,row,profile):
        from PIL import Image
        started=perf_counter(); t=perf_counter(); rgb=np.asarray(Image.open(row['image_path']).convert('RGB')); decode=(perf_counter()-t)*1000
        t=perf_counter(); det_input,ratio=self.detector.preprocess(image); prep_det=(perf_counter()-t)*1000
        t=perf_counter(); det_outputs=self.detector.inference(det_input); inf_det=(perf_counter()-t)*1000
        t=perf_counter(); boxes=self.detector.postprocess(det_outputs[0],ratio); post_det=(perf_counter()-t)*1000
        if len(boxes)==0:
            timing={'decode':decode,'preprocess':prep_det,'inference':inf_det,'postprocess':post_det,'end_to_end':(perf_counter()-started)*1000}
            return Prediction(row['sample_id'],row['dataset_id'],self.model_id,'','no_detection',timings_ms=timing,error_code='NO_DETECTION',native={'detections':0})
        areas=(boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1]); box=boxes[int(np.argmax(areas))]
        t=perf_counter(); pose_input,center,scale=self.pose.preprocess(image,box); prep_pose=(perf_counter()-t)*1000
        t=perf_counter(); pose_outputs=self.pose.inference(pose_input); inf_pose=(perf_counter()-t)*1000
        t=perf_counter(); points,scores=self.pose.postprocess(pose_outputs,center,scale); post_pose=(perf_counter()-t)*1000
        points,scores=np.asarray(points),np.asarray(scores)
        if points.ndim==3 and points.shape[0]==1: points=points[0]
        if scores.ndim==2 and scores.shape[0]==1: scores=scores[0]
        timing={'decode':decode,'preprocess':prep_det+prep_pose,'inference':inf_det+inf_pose,'postprocess':post_det+post_pose,'end_to_end':(perf_counter()-started)*1000}
        if points.shape!=(17,2) or scores.shape!=(17,):
            return Prediction(row['sample_id'],row['dataset_id'],self.model_id,'','mapping_error',timings_ms=timing,error_code='INVALID_COCO17_OUTPUT',native={'points_shape':list(points.shape),'scores_shape':list(scores.shape),'selected_bbox':np.asarray(box).tolist(),'detected_people':int(len(boxes))})
        pts=points.astype(float).tolist(); confidence=scores.astype(float).tolist()
        return Prediction(row['sample_id'],row['dataset_id'],self.model_id,'','ok',pts,confidence,timings_ms=timing,native={'selected_bbox':np.asarray(box).tolist(),'detected_people':int(len(boxes))})
    def provenance(self): return {'model_id':self.model_id,'adapter':type(self).__name__,'weight_source':'OpenMMLab RTMlib ONNX SDK (RTMPose-M + YOLOX-M)','runtime_device':getattr(self,'runtime_device','unknown'),'amp':False,'compile':False,'quantization':False,'tensorrt':False}
