from .base import BaseAdapter
from benchmark_core.contracts import Prediction
class MediaPipeHeavy(BaseAdapter):
    model_id='mediapipe_pose_heavy'; MAP=(0,2,5,7,8,11,12,13,14,15,16,23,24,25,26,27,28)
    def load(self,profile):
        import mediapipe as mp; self.mp=mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        delegate=getattr(python.BaseOptions,'Delegate',None)
        self.gpu_requested=bool(delegate and hasattr(delegate,'GPU'))
        try:
            options=python.BaseOptions(model_asset_path=self.model_path,delegate=delegate.GPU) if self.gpu_requested else python.BaseOptions(model_asset_path=self.model_path)
            self.landmarker=vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(base_options=options,running_mode=vision.RunningMode.IMAGE,num_poses=1))
            self.runtime_device='gpu' if self.gpu_requested else 'cpu'
        except Exception as exc:
            self.gpu_fallback_reason=f'{type(exc).__name__}: {exc}'
            self.landmarker=vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=self.model_path),running_mode=vision.RunningMode.IMAGE,num_poses=1))
            self.runtime_device='cpu'
    def predict(self,row,profile):
        import cv2
        from time import perf_counter
        started=perf_counter(); t=perf_counter(); image=cv2.imread(row['image_path'])
        if image is None: return Prediction(row['sample_id'],row['dataset_id'],self.model_id,'','input_error',timings_ms={'decode':0.,'preprocess':0.,'inference':0.,'postprocess':0.,'end_to_end':(perf_counter()-started)*1000},error_code='IMAGE_READ_FAILED')
        decode=(perf_counter()-t)*1000; t=perf_counter(); rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB); mp_image=self.mp.Image(image_format=self.mp.ImageFormat.SRGB,data=rgb); prep=(perf_counter()-t)*1000
        result,inf=self.clock(lambda:self.landmarker.detect(mp_image)); t=perf_counter()
        timings={'decode':decode,'preprocess':prep,'inference':inf,'postprocess':0.,'end_to_end':0.}
        if not result.pose_landmarks:
            timings['postprocess']=(perf_counter()-t)*1000; timings['end_to_end']=(perf_counter()-started)*1000
            return Prediction(row['sample_id'],row['dataset_id'],self.model_id,'','no_detection',timings_ms=timings,error_code='NO_DETECTION',native={'pose_landmarks':[]})
        h,w=rgb.shape[:2]; lm=result.pose_landmarks[0]; pts=[[lm[i].x*w,lm[i].y*h] for i in self.MAP]; scores=[float(min(getattr(lm[i],'visibility',0.),getattr(lm[i],'presence',0.))) for i in self.MAP]
        native={'pose_landmarks':[{'x':float(x.x),'y':float(x.y),'z':float(x.z),'visibility':float(x.visibility),'presence':float(x.presence)} for x in lm]}
        timings['postprocess']=(perf_counter()-t)*1000; timings['end_to_end']=(perf_counter()-started)*1000
        return Prediction(row['sample_id'],row['dataset_id'],self.model_id,'','ok',pts,scores,timings_ms=timings,native=native)

    def provenance(self):
        return {'model_id':self.model_id,'adapter':type(self).__name__,'weight_source':'MediaPipe official pose_landmarker_heavy.task','runtime_device':getattr(self,'runtime_device','unknown'),'gpu_fallback_reason':getattr(self,'gpu_fallback_reason',None),'amp':False,'compile':False,'quantization':False,'tensorrt':False}
