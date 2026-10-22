from time import perf_counter
import numpy as np
from .base import BaseAdapter
from benchmark_core.contracts import Prediction

class MoveNetThunder(BaseAdapter):
    model_id='movenet_thunder'
    def __init__(self, model_path): self.model_path=str(model_path)
    def load(self, profile):
        import tensorflow as tf
        self.tf=tf; self.gpu_delegate=None
        if __import__('os').environ.get('INFANTPOSE_MOVENET_GPU','1')=='1':
            try:
                self.gpu_delegate=tf.lite.experimental.load_delegate('libtensorflowlite_gpu_delegate.so')
            except Exception as exc: self.gpu_fallback_reason=f'{type(exc).__name__}: {exc}'
        delegates=[self.gpu_delegate] if self.gpu_delegate else None
        self.interpreter=tf.lite.Interpreter(model_path=self.model_path,num_threads=1,experimental_delegates=delegates); self.interpreter.allocate_tensors()
        self.runtime_device='gpu_delegate' if self.gpu_delegate else 'cpu'
        self.input=self.interpreter.get_input_details()[0]; self.output=self.interpreter.get_output_details()[0]
    def predict(self,row,profile):
        from PIL import Image
        started=perf_counter(); t=perf_counter(); rgb=np.asarray(Image.open(row['image_path']).convert('RGB')); decode=(perf_counter()-t)*1000
        t=perf_counter(); x=self.tf.image.resize_with_pad(self.tf.convert_to_tensor(rgb),256,256); x=self.tf.cast(self.tf.expand_dims(x,0),self.input['dtype']); prep=(perf_counter()-t)*1000
        t=perf_counter(); self.interpreter.set_tensor(self.input['index'],x.numpy()); self.interpreter.invoke(); raw=self.interpreter.get_tensor(self.output['index'])[0,0]; inf=(perf_counter()-t)*1000
        t=perf_counter(); h,w=rgb.shape[:2]; pts=[[float(k[1]*w),float(k[0]*h)] for k in raw]; scores=[float(k[2]) for k in raw]; status='ok' if max(scores)>=.05 else 'no_detection'; post=(perf_counter()-t)*1000
        timing={'decode':decode,'preprocess':prep,'inference':inf,'postprocess':post,'end_to_end':(perf_counter()-started)*1000}
        return Prediction(row['sample_id'],row['dataset_id'],self.model_id,'',status,pts,scores,timings_ms=timing,error_code=None if status=='ok' else 'NO_DETECTION',native={'movenet_output_yxs':raw.tolist()})
    def provenance(self): return {'model_id':self.model_id,'adapter':type(self).__name__,'weight_source':'TF Hub MoveNet Thunder TFLite','runtime_device':getattr(self,'runtime_device','unknown'),'gpu_fallback_reason':getattr(self,'gpu_fallback_reason',None),'amp':False,'compile':False,'quantization':False,'tensorrt':False,'num_threads':1}
