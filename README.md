This project implements GPU _NVIDIA CUDA_ inference support for well known [Tencent NCNN](https://github.com/Tencent/ncnn) inference engine. Many of the Edge AI projects on NVIDIA Jetson family of devices could benefit from this support.

---

### Development Status

Following layers have been currently implemented in CUDA: _AbsVal, BatchNorm, Bias, BinaryOp, BNLL, Concat, Convolution, ConvolutionDepthWise, Crop, Flatten, InnerProduct, Input, Packing, Padding, Quantize, ReLU, Reshape, Softmax, Split_

Development plan for the near future:
* Cuda implementation of layers _Pooling, Eltwise, HardSigmoid, HardSwish, Interp, Scale, Yolov3DetectionOutput_
* Further optimization of existing CUDA layers (with the goal to beat Vulkan performance ;) )

For usecases where some CUDA layer implementation is missing, CPU/GPU data ping-pong will slow the execution significantly.

_Develop_ branch is used for active development. Development of new layers is performed on develop_<layer_name> branch which is squashed before merging to develop branch. Occasionaly upstream updates and fixes would be added to the project.

### Build and test

#### Build project:

```
git clone https://github.com/atanmarko/ncnn-with-cuda
cd ncnn-with-cuda
mkdir build
cd build
cmake -DNCNN_VULKAN=OFF -DNCNN_CUDA=ON -DLOG_LAYERS=OFF ..
make
```

#### Test particular layer

To run test for particular layer, where CPU vs CUDA implementation and execution speed is tested (<layer_name> is name of the CUDA layer in small case, e.g. _test_convolution_):
```
cd build/tests
./test_<layer_name>
```

#### Check which layers are not executed on CUDA

Build project with turned on LOG_LAYERS config parameter:
```
cmake -DNCNN_VULKAN=OFF -DNCNN_CUDA=ON -DLOG_LAYERS=ON ..
```
Run the particular example or network benchmark and grep for non cuda layers:
```
./retinaface <path to image file> | grep forward | grep -v cuda
```



#### Run retinaface test program:

Copy _mnet.25-opt.bin_ and _mnet.25-opt.param_  files to the build/examples directory [available here](https://github.com/nihui/ncnn-assets.git): 

```
cd build/examples
./retinaface <path_to_picture_file>
```

### License

NCNN CUDA implementation:
[BSD 3 Clause](LICENSE.txt)

Original NCNN Licence:
[BSD 3 Clause](https://github.com/Tencent/ncnn/blob/master/LICENSE.txt)

