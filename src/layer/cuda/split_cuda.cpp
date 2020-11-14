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


#include "split_cuda.h"


namespace ncnn {


Split_cuda::Split_cuda()
{
    support_cuda = true;
}


int Split_cuda::forward(const std::vector<CudaMat>& bottom_blobs, std::vector<CudaMat>& top_blobs, const Option& /*opt*/) const
{
    const CudaMat& bottom_blob = bottom_blobs[0];
    for (size_t i = 0; i < top_blobs.size(); i++)
    {
        top_blobs[i] = bottom_blob;
    }

    return 0;
}


} // namespace ncnn
