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

Benchmark Retinaface:

Copy _mnet.25-opt.bin_ and _mnet.25-opt.param_  files to the build/benchmark directory.

```
cd build/benchmark
./retinaface-benchmark <path_to_picture_file>
```
It will run 10 loops of Retinaface face detection and print inference timing results. Retinaface stride 32 has all the layers implemented in CUDA.

| | Image Size | Stride | CPU av. time (us) | Vulkan av. time (us) | CUDA av. time (us)
------------ | ------------- |  ------------- |  ------------- | -------------  | -------------
 i7-4790, GTX 1060  | 640x480 | 32 | 28.90 | 33.20  | 31.10
 i7-4790, GTX 1060  | 1280x720 | 32 | 92.70 | 96.90  | 54.00
 i7-4790, GTX 1060  | 1920x1080 | 32 |  167.50 | 204.50  | 91.70
 Jetson AGX Xavier  | 640x480 | 32 | 373.20 | 402.10  | 343.60
 Jetson AGX Xavier  | 1280x720 | 32 | 508.30 | 738.40  | 327.60
 Jetson AGX Xavier  | 1920x1080 | 32 |  812.00 | 934.70  | 436.70
 



### License

NCNN CUDA implementation:
[BSD 3 Clause](LICENSE.txt)

Original NCNN Licence:
[BSD 3 Clause](https://github.com/Tencent/ncnn/blob/master/LICENSE.txt)

