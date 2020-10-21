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


#include "packing_cuda.h"

#include <math.h>

namespace ncnn {

int packing_cuda_forward(const CudaMat& bottom_blob, CudaMat& top_blob, const Packing_cuda::packing_options options);

Packing_cuda::Packing_cuda()
{
    support_cuda = true;
}


int Packing_cuda::load_param(const ParamDict& pd)
{
    Packing::load_param(pd);

    return 0;
}


int Packing_cuda::forward(const CudaMat& bottom_blob, CudaMat& top_blob, const Option&) const
{
    return packing_cuda_forward(bottom_blob,
                                top_blob,
                                Packing_cuda::packing_options{*this});
}


} // namespace ncnn
