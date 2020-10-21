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


#include "bias_cuda.h"

#include <algorithm>
#include <chrono>

namespace ncnn {

int bias_cuda_forward_inplace(float* a_input, const ncnn::CudaMatInfo& a_info, const float* bias);

Bias_cuda::Bias_cuda()
{
    support_cuda = true;
}


int Bias_cuda::load_model(const CudaModelBinFromMatArray& mb)
{
    if (!this->support_cuda)
        return -100;


    std::shared_ptr<ncnn::CudaAllocator> cuda_allocator = ncnn::get_current_gpu_allocator();

    bias_data = CudaMat{mb.load(bias_data_size, 1), cuda_allocator};
    if (bias_data.empty())
        return -100;

    return 0;
}

int Bias_cuda::forward_inplace(CudaMat& bottom_top_blob, const Option& opt __attribute__((unused))) const
{
#if LOG_LAYERS
    LOGL("Bias_cuda forward_inplace");
#endif
    CudaMatInfo a_info{bottom_top_blob};

    return bias_cuda_forward_inplace(static_cast<float*>(bottom_top_blob.get_raw_data()), a_info,
                              static_cast<const float*>(bias_data.get_craw_data()));
}

} // namespace ncnn
