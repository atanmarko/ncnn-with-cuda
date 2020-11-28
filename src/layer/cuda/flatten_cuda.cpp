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


#include "flatten_cuda.h"

#include <chrono>

namespace ncnn {

int flatten_cuda_forward(const unsigned char* bottom_blob, const ncnn::CudaMatInfo bottom_blob_info,
                         unsigned char* top_blob);

Flatten_cuda::Flatten_cuda()
{
    support_cuda = true;
}

int Flatten_cuda::forward(const CudaMat& bottom_blob, CudaMat& top_blob, const Option&) const
{

    CudaMatInfo bottom_blob_info{bottom_blob};

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;
    int size = w * h;

    std::shared_ptr<ncnn::CudaAllocator> cuda_allocator = ncnn::get_current_gpu_allocator();
    top_blob.create(size * channels, elemsize, cuda_allocator);
    if (top_blob.empty())
        return -100;


    return flatten_cuda_forward(
        static_cast<const unsigned char*>(bottom_blob.get_craw_data()),
        bottom_blob_info,
        static_cast<unsigned char*>(top_blob.get_raw_data())
        );
}

} // namespace ncnn
