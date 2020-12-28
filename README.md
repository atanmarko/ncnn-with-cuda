This project implements GPU _NVIDIA CUDA_ inference support for well known [Tencent NCNN](https://github.com/Tencent/ncnn) inference engine. Many of the Edge AI projects on NVIDIA Jetson family of devices could benefit from this support.

---

### Development Status

Following layers have been currently implemented in CUDA: _AbsVal, BatchNorm, Bias, BinaryOp, BNLL, Convolution, ConvolutionDepthWise, Crop, Flatten, InnerProduct, Input, Packing, Padding, Quantize, ReLU, Reshape, Softmax, Split_.

Next for development on TODO list: _Concat, Interp, Scale, Pooling, Yolov3DetectionOutput_

For usecases where some CUDA layer implementation is missing, CPU/GPU data ping-pong will slow the execution significantly. _RetinaFace_ face detection currently only misses _Concat_ and _Interp_ layers in order to be fully executable on GPU. 

_Develop_ branch is used for active development. Development of new layers is performed on develop_<layer_name> branch. Occasionaly upstream updates and fixes would be added to the project.

### Build and test

#### Build project:

```
git clone https://github.com/atanmarko/ncnn-with-cuda
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

