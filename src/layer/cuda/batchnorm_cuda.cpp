//
// Author: Marko Atanasievski
//
// Copyright (C) 2020 TANCOM SOFTWARE SOLUTIONS Ltd. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.
//
// Parts of this file are originally copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.


#include "batchnorm_cuda.h"

namespace ncnn {

//void batchnorm_cuda_load_model(int channels, float eps, float* a_data_gpu, float* b_data_gpu,
//                                         float* bias_data_gpu, float* slope_data_gpu, float* mean_data_gpu, float* var_data_gpu);
int batchnorm_cuda_forward_inplace(float* d_input, const float* b_data_gpu, const float* a_data_gpu, const CudaMatInfo& matInfo);

BatchNorm_cuda::BatchNorm_cuda()
{
    support_cuda = true;
    _cuda_allocator = ncnn::get_current_gpu_allocator();
}

int BatchNorm_cuda::create_pipeline(const Option& opt)
{
    BatchNorm::create_pipeline(opt);

    return 0;
}

int BatchNorm_cuda::destroy_pipeline(const Option& opt)
{
    BatchNorm::destroy_pipeline(opt);
    return 0;
}


int BatchNorm_cuda::load_model(const CudaModelBinFromMatArray& mb)
{
    BatchNorm::load_model(mb);

    if (!this->support_cuda)
        return -100;

    slope_data_gpu = CudaMat{slope_data, _cuda_allocator};
    if (slope_data_gpu.empty())
        return -100;

    mean_data_gpu = CudaMat{mean_data, _cuda_allocator};
    if (mean_data_gpu.empty())
        return -100;

    var_data_gpu = CudaMat{var_data, _cuda_allocator};
    if (var_data_gpu.empty())
        return -100;

    bias_data_gpu = CudaMat{bias_data, _cuda_allocator};
    if (bias_data_gpu.empty())
        return -100;

    a_data_gpu = CudaMat{a_data, _cuda_allocator};
    if (a_data_gpu.empty())
        return -100;

    b_data_gpu = CudaMat{b_data, _cuda_allocator};
    if (b_data_gpu.empty())
        return -100;

    return 0;
}

int BatchNorm_cuda::load_model(const ModelBin& mb)
{

    BatchNorm::load_model(mb);

    if (!this->support_cuda)
        return -100;

    slope_data_gpu = CudaMat{slope_data, _cuda_allocator};
    if (slope_data_gpu.empty())
        return -100;

    mean_data_gpu = CudaMat{mean_data, _cuda_allocator};
    if (mean_data_gpu.empty())
        return -100;

    var_data_gpu = CudaMat{var_data, _cuda_allocator};
    if (var_data_gpu.empty())
        return -100;

    bias_data_gpu = CudaMat{bias_data, _cuda_allocator};
    if (bias_data_gpu.empty())
        return -100;

    a_data_gpu = CudaMat{a_data, _cuda_allocator};
    if (a_data_gpu.empty())
        return -100;

    b_data_gpu = CudaMat{b_data, _cuda_allocator};
    if (b_data_gpu.empty())
        return -100;

    return 0;
}


int BatchNorm_cuda::forward_inplace(CudaMat& bottom_top_blob, const Option& /*opt*/) const
{
#if LOG_LAYERS
    LOGL("BatchNorm_cuda forward_inplace");
#endif
    return batchnorm_cuda_forward_inplace(static_cast<float*>(bottom_top_blob.get_raw_data()),
                                   static_cast<const float*>(b_data_gpu.get_raw_data()),
                                   static_cast<const float*>(a_data_gpu.get_raw_data()),
                                   CudaMatInfo{bottom_top_blob});
}

} // namespace ncnn
