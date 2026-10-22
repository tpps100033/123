# OpenPose runtime

The source Notebook uses a cached Ubuntu build of OpenPose v1.7.0 and a COCO-18 Caffe model. It is not a Python wheel dependency. Its CUDA/cuDNN/CMake/compiler build must be validated against RTX 5060 (SM 12.0) under WSL2 before it can be called from the local adapter. Until that validation exists, keep the source Notebook and cached artifacts untouched and report OpenPose as blocked, never as passed.
