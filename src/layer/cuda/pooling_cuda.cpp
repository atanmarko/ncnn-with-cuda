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


#include "layer_type.h"
#include "pooling_cuda.h"

#include <chrono>
#include <float.h>

namespace ncnn {

int pooling_cuda_forward_global(const CudaMat& bottom_blob, CudaMat& top_blob, const Pooling_cuda::Pooling_info& info);
int pooling_cuda_forward(const CudaMat& bottom_blob, CudaMat& top_blob, const Pooling_cuda::Pooling_info& info);

Pooling_cuda::Pooling_cuda()
{
    support_cuda = true;

    int current_state = 0;
    checkCudaErrors(cudaMalloc((void**)&pooling_lock, sizeof(int)));
    checkCudaErrors(cudaMemcpy(pooling_lock, &current_state, sizeof(int), cudaMemcpyHostToDevice));
}

Pooling_cuda::~Pooling_cuda() {
    checkCudaErrors(cudaFree(pooling_lock));

}


void Pooling_cuda::make_padding(const CudaMat& bottom_blob, CudaMat& bottom_blob_bordered, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;

    bottom_blob_bordered = bottom_blob;

    float pad_value = 0.f;
    if (pooling_type == PoolMethod_MAX)
    {
        pad_value = bottom_blob.elemsize == 1 ? -128.f : -FLT_MAX;
    }
    else if (pooling_type == PoolMethod_AVE)
    {
        pad_value = 0.f;
    }

    int wtailpad = 0;
    int htailpad = 0;

    if (pad_mode == 0) // full padding
    {
        int wtail = (w + pad_left + pad_right - kernel_w) % stride_w;
        int htail = (h + pad_top + pad_bottom - kernel_h) % stride_h;

        if (wtail != 0)
            wtailpad = stride_w - wtail;
        if (htail != 0)
            htailpad = stride_h - htail;

        Option opt_b = opt;
        opt_b.blob_allocator = opt.workspace_allocator;
        opt_b.blob_cuda_allocator = opt.blob_cuda_allocator;
        copy_make_border(bottom_blob, bottom_blob_bordered, pad_top, pad_bottom + htailpad, pad_left, pad_right + wtailpad, BORDER_CONSTANT, pad_value, opt_b);
    }
    else if (pad_mode == 1) // valid padding
    {
        Option opt_b = opt;
        opt_b.blob_allocator = opt.workspace_allocator;
        opt_b.blob_cuda_allocator = opt.blob_cuda_allocator;
        copy_make_border(bottom_blob, bottom_blob_bordered, pad_top, pad_bottom, pad_left, pad_right, BORDER_CONSTANT, pad_value, opt_b);
    }
    else if (pad_mode == 2) // tensorflow padding=SAME or onnx padding=SAME_UPPER
    {
        int wpad = kernel_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_h + (h - 1) / stride_h * stride_h - h;
        if (wpad > 0 || hpad > 0)
        {
            Option opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            opt_b.blob_cuda_allocator = opt.blob_cuda_allocator;
            copy_make_border(bottom_blob, bottom_blob_bordered, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, BORDER_CONSTANT, pad_value, opt_b);
        }
    }
    else if (pad_mode == 3) // onnx padding=SAME_LOWER
    {
        int wpad = kernel_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_h + (h - 1) / stride_h * stride_h - h;
        if (wpad > 0 || hpad > 0)
        {
            Option opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            opt_b.blob_cuda_allocator = opt.blob_cuda_allocator;
            copy_make_border(bottom_blob, bottom_blob_bordered, hpad - hpad / 2, hpad / 2, wpad - wpad / 2, wpad / 2, BORDER_CONSTANT, pad_value, opt_b);
        }
    }

}


int Pooling_cuda::forward(const CudaMat& bottom_blob, CudaMat& top_blob, const Option& opt) const
{
#if LOG_LAYERS
    LOGL("Pooling_cuda forward");
#endif

    // max value in NxN window
    // avg value in NxN window

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

    const int maxk = kernel_w * kernel_h;


    //     NCNN_LOGE("Pooling     input %d x %d  pad = %d %d %d %d  ksize=%d %d  stride=%d %d", w, h, pad_left, pad_right, pad_top, pad_bottom, kernel_w, kernel_h, stride_w, stride_h);
    if (global_pooling)
    {
        top_blob.create(channels, elemsize, opt.blob_cuda_allocator);
        if (top_blob.empty())
            return -100;

        return pooling_cuda_forward_global(bottom_blob, top_blob, Pooling_info(*this, std::vector<int>{}));
    }

    CudaMat bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;

    int outw = (w - kernel_w) / stride_w + 1;
    int outh = (h - kernel_h) / stride_h + 1;

    top_blob.create(outw, outh, channels, elemsize, opt.blob_cuda_allocator);
    if (top_blob.empty())
        return -100;


    // kernel offsets
    std::vector<int> _space_ofs = std::vector<int>(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w - kernel_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2++;
            }
            p2 += gap;
        }
    }

    Pooling_info info{*this, _space_ofs};

    if (avgpool_count_include_pad == 0)
    {
        int wtailpad = 0;
        int htailpad = 0;

        if (pad_mode == 0) // full padding
        {
            wtailpad = bottom_blob_bordered.w - bottom_blob.w - pad_left - pad_right;
            htailpad = bottom_blob_bordered.h - bottom_blob.h - pad_top - pad_bottom;
        }

        info.wtailpad = wtailpad;
        info.htailpad = htailpad;
    }

    return pooling_cuda_forward(bottom_blob_bordered, top_blob, info);

}

} // namespace ncnn
