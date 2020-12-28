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


#include "relu_cuda.h"

#include <algorithm>
#include <chrono>

namespace ncnn {

int relu_cuda_forward_inplace(float* d_input, int input_size, float slope);
int relu_cuda_forward_inplace_int8(int8_t * d_input, int input_size, float slope);

ReLU_cuda::ReLU_cuda()
{
    support_cuda = true;
}


int ReLU_cuda::forward_inplace_int8(CudaMat& bottom_top_blob, const Option& /*opt*/) const
{
#if LOG_LAYERS
    LOGL("ReLU_cuda forward_inplace_int8");
#endif

    const int total_size = bottom_top_blob.total();
    relu_cuda_forward_inplace_int8(static_cast<int8_t*>(bottom_top_blob.get_raw_data()), total_size, slope);

    return 0;
}

int ReLU_cuda::forward_inplace(CudaMat& bottom_top_blob, const Option& opt) const
{
#if LOG_LAYERS
    LOGL("ReLU_cuda forward_inplace");
#endif
    if (bottom_top_blob.elemsize == 1u)
        return ReLU_cuda::forward_inplace_int8(bottom_top_blob, opt);

    const int total_size = bottom_top_blob.total();
    relu_cuda_forward_inplace(static_cast<float*>(bottom_top_blob.get_raw_data()), total_size, slope);

    return 0;
}

} // namespace ncnn
