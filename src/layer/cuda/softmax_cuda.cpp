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


#include "softmax_cuda.h"

#include <math.h>
#include <vector>

namespace ncnn {

int softmax_cuda_forward_inplace(float* a_input, const ncnn::CudaMatInfo& a_info,
                                 const int axis,
                                 float* gpu_scratchpad_memory,
                                 int gpu_scratchpad_memory_size);

Softmax_cuda::Softmax_cuda()
{
    support_cuda = true;

    std::shared_ptr<ncnn::CudaAllocator> cuda_allocator = ncnn::get_current_gpu_allocator();
    gpu_scratch_pad_memory = static_cast<float *>(cuda_allocator->fastMalloc(gpu_scratch_pad_memory_size));
}

Softmax_cuda::~Softmax_cuda()
{
    std::shared_ptr<ncnn::CudaAllocator> cuda_allocator = ncnn::get_current_gpu_allocator();
    cuda_allocator->fastFree(gpu_scratch_pad_memory);
}


int Softmax_cuda::load_param(const ParamDict& pd)
{
    return Softmax::load_param(pd);
}



int Softmax_cuda::forward_inplace(CudaMat& bottom_top_blob, const Option&) const
{
#if LOG_LAYERS
    LOGL("Softmax_cuda forward_inplace");
#endif
    return softmax_cuda_forward_inplace(static_cast<float*>(bottom_top_blob.get_raw_data()),
                                        CudaMatInfo{bottom_top_blob},
                                        axis,
                                        gpu_scratch_pad_memory,
                                        gpu_scratch_pad_memory_size);
}



} // namespace ncnn
