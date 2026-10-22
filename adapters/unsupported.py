from .base import BaseAdapter
class InstallRequiredAdapter(BaseAdapter):
    def __init__(self,model_id,reason): self.model_id=model_id; self.reason=reason
    def predict(self,row,profile): raise RuntimeError(self.reason)
