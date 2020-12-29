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


#include "concat_cuda.h"

#include <chrono>

namespace ncnn {

int concat_cuda_forward(const std::vector<CudaMat>& bottom_blobs, CudaMat& top_blob, const int axis);

Concat_cuda::Concat_cuda()
{
    support_cuda = true;
}

int Concat_cuda::forward(const std::vector<CudaMat>& bottom_blobs, std::vector<CudaMat>& top_blobs, const Option&) const
{
#if LOG_LAYERS
    LOGL("Concat_cuda forward");
#endif

    int dims = bottom_blobs[0].dims;
    size_t elemsize = bottom_blobs[0].elemsize;
    int positive_axis = axis < 0 ? dims + axis : axis;

    std::shared_ptr<ncnn::CudaAllocator> cuda_allocator = ncnn::get_current_gpu_allocator();

    if (dims == 1) // positive_axis == 0
    {
        // concat vector
        // total length
        int top_w = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const CudaMat& bottom_blob = bottom_blobs[b];
            top_w += bottom_blob.w;
        }

        CudaMat& top_blob = top_blobs[0];
        top_blob.create(top_w, elemsize, cuda_allocator);
        if (top_blob.empty())
            return -100;


        return concat_cuda_forward(bottom_blobs, top_blob, positive_axis);
    }

    if (dims == 2 && positive_axis == 0)
    {
        // concat image
        int w = bottom_blobs[0].w;

        // total height
        int top_h = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const CudaMat& bottom_blob = bottom_blobs[b];
            top_h += bottom_blob.h;
        }

        CudaMat& top_blob = top_blobs[0];
        top_blob.create(w, top_h, elemsize, cuda_allocator);
        if (top_blob.empty())
            return -100;

        return concat_cuda_forward(bottom_blobs, top_blob, positive_axis);
    }

    if (dims == 2 && positive_axis == 1)
    {
        // interleave image row
        int h = bottom_blobs[0].h;

        // total width
        int top_w = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const CudaMat& bottom_blob = bottom_blobs[b];
            top_w += bottom_blob.w;
        }

        CudaMat& top_blob = top_blobs[0];
        top_blob.create(top_w, h, elemsize, cuda_allocator);
        if (top_blob.empty())
            return -100;

        return concat_cuda_forward(bottom_blobs, top_blob, positive_axis);
    }


    if (dims == 3 && positive_axis == 1)
    {
        // interleave dim height
        int w = bottom_blobs[0].w;
        int channels = bottom_blobs[0].c;

        // total height
        int top_h = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const CudaMat& bottom_blob = bottom_blobs[b];
            top_h += bottom_blob.h;
        }

        CudaMat& top_blob = top_blobs[0];
        top_blob.create(w, top_h, channels, elemsize, cuda_allocator);
        if (top_blob.empty())
            return -100;

        return concat_cuda_forward(bottom_blobs, top_blob, positive_axis);
    }


    if (dims == 3 && positive_axis == 2)
    {
        // interleave dim width
        int h = bottom_blobs[0].h;
        int channels = bottom_blobs[0].c;

        // total height
        int top_w = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const CudaMat& bottom_blob = bottom_blobs[b];
            top_w += bottom_blob.w;
        }

        CudaMat& top_blob = top_blobs[0];
        top_blob.create(top_w, h, channels, elemsize, cuda_allocator);
        if (top_blob.empty())
            return -100;

        return concat_cuda_forward(bottom_blobs, top_blob, positive_axis);
    }

    if (dims == 3 && positive_axis == 0)
    {
        // concat dim
        int w = bottom_blobs[0].w;
        int h = bottom_blobs[0].h;

        // total channels
        int top_channels = 0;
        for (size_t b = 0; b < bottom_blobs.size(); b++)
        {
            const CudaMat& bottom_blob = bottom_blobs[b];
            top_channels += bottom_blob.c;
        }

        CudaMat& top_blob = top_blobs[0];
        top_blob.create(w, h, top_channels, elemsize, cuda_allocator);
        if (top_blob.empty())
            return -100;

        return concat_cuda_forward(bottom_blobs, top_blob, positive_axis);
    }




    return 0;
}

} // namespace ncnn
