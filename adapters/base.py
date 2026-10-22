from abc import ABC, abstractmethod
from time import perf_counter
class BaseAdapter(ABC):
    model_id='unset'
    def load(self,profile): pass
    def warmup(self,rows,profile):
        self.warmup_errors=[]
        for row in rows:
            try: self.predict(row,profile)
            except Exception as exc:
                self.warmup_errors.append(type(exc).__name__)
                try:
                    import torch
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                except Exception: pass
    def predict_batch(self,rows,profile):
        """Default batch interface preserves each model's single-image semantics."""
        return [self.predict(row,profile) for row in rows]
    @abstractmethod
    def predict(self,row,profile): ...
    def provenance(self): return {'model_id':self.model_id,'adapter':type(self).__name__}
    @staticmethod
    def clock(fn):
        t=perf_counter(); result=fn(); return result,(perf_counter()-t)*1000
