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


#include "reshape_cuda.h"

#include <chrono>

namespace ncnn {

int reshape_cuda_forward(const ncnn::CudaMat& bottom_blob, ncnn::CudaMat& top_blob, const int outw, const int outh, const int outc, bool need_permute, int ndim, const Option& opt);


Reshape_cuda::Reshape_cuda()
{
    support_cuda = true;
}

int Reshape_cuda::forward(const CudaMat& bottom_blob, CudaMat& top_blob, const Option& opt) const
{
#if LOG_LAYERS
    LOGL("Reshape_cuda forward");
#endif
    int total = bottom_blob.w * bottom_blob.h * bottom_blob.c;

    int dims = bottom_blob.dims;

    // resolve out shape
    int outw = w;
    int outh = h;
    int outc = c;

    if (ndim == 1)
    {
        if (outw == 0)
            outw = bottom_blob.w;

        if (outw == -1)
            outw = total;

        if (dims == 1 && bottom_blob.w == outw)
        {
            top_blob = bottom_blob;
            return 0;
        }
    }
    if (ndim == 2)
    {
        if (outw == 0)
            outw = bottom_blob.w;
        if (outh == 0)
            outh = bottom_blob.h;

        if (outw == -1)
            outw = total / outh;
        if (outh == -1)
            outh = total / outw;

        if (dims == 2 && bottom_blob.h == outh)
        {
            top_blob = bottom_blob;
            return 0;
        }
    }
    if (ndim == 3)
    {
        if (outw == 0)
            outw = bottom_blob.w;
        if (outh == 0)
            outh = bottom_blob.h;
        if (outc == 0)
            outc = bottom_blob.c;

        if (outw == -1)
            outw = total / outc / outh;
        if (outh == -1)
            outh = total / outc / outw;
        if (outc == -1)
            outc = total / outh / outw;

        if (dims == 3 && bottom_blob.c == outc)
        {
            top_blob = bottom_blob;
            top_blob.w = outw;
            top_blob.h = outh;
            return 0;
        }
    }

    bool need_permute = permute == 1;
    if (dims == 2 && ndim == 2 && bottom_blob.h == outh)
        need_permute = false;
    if (dims == 3 && ndim == 3 && bottom_blob.c == outc)
        need_permute = false;


    return reshape_cuda_forward(bottom_blob, top_blob, outw, outh, outc, need_permute, ndim, opt);




}

} // namespace ncnn
