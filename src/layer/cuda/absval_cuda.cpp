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


#include "absval_cuda.h"

namespace ncnn {

int relu_absval_forward_inplace(float * d_input, const int input_size);

AbsVal_cuda::AbsVal_cuda()
{
    support_cuda = true;
}

int AbsVal_cuda::forward_inplace(CudaMat& bottom_top_blob, const Option& /*opt*/) const
{
    const int total_size = bottom_top_blob.total();

    relu_absval_forward_inplace(static_cast<float*>(bottom_top_blob.get_raw_data()), total_size);

    return 0;
}

} // namespace ncnn
