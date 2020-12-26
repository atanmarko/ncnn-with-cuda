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
#include "convolutiondepthwise_cuda.h"

#include <chrono>

namespace ncnn {

int convolutiondepthwise_cuda_forward(const CudaMat& bottom_blob, CudaMat& top_blob, const ConvolutionDepthWise_cuda::ConvolutionDepthWise_info& info);
int convolutiondepthwise_cuda_forward_int8(const CudaMat& bottom_blob, CudaMat& top_blob, const ConvolutionDepthWise_cuda::ConvolutionDepthWise_info& info);

ConvolutionDepthWise_cuda::ConvolutionDepthWise_cuda()
{
    support_cuda = true;
    std::shared_ptr<ncnn::CudaAllocator> cuda_allocator = ncnn::get_current_gpu_allocator();
    gpu_top_blob_int8_scale = static_cast<float *>(cuda_allocator->fastMalloc(sizeof(float)));
}

ConvolutionDepthWise_cuda::~ConvolutionDepthWise_cuda()
{
    std::shared_ptr<ncnn::CudaAllocator> cuda_allocator = ncnn::get_current_gpu_allocator();
    cuda_allocator->fastFree(gpu_top_blob_int8_scale);
}

int ConvolutionDepthWise_cuda::load_param(const ParamDict& pd)
{
    ConvolutionDepthWise::load_param(pd);

    std::shared_ptr<ncnn::CudaAllocator> cuda_allocator = ncnn::get_current_gpu_allocator();
    gpu_activation_params = CudaMat{activation_params, cuda_allocator};

    return 0;
}

int ConvolutionDepthWise_cuda::load_model(const CudaModelBinFromMatArray& mb)
{
    int result = ConvolutionDepthWise::load_model(static_cast<ModelBinFromMatArray>(mb));
    if (result < 0)
        return result;

    std::shared_ptr<ncnn::CudaAllocator> cuda_allocator = ncnn::get_current_gpu_allocator();

    gpu_weight_data = CudaMat{weight_data, cuda_allocator};

    if (bias_term)
    {
        gpu_bias_data = CudaMat{bias_data, cuda_allocator};
    }

    if (int8_scale_term)
    {
        gpu_weight_data_int8_scales = CudaMat{weight_data_int8_scales, cuda_allocator};
        gpu_bottom_blob_int8_scales = CudaMat{bottom_blob_int8_scales, cuda_allocator};
        checkCudaErrors(cudaMemcpy(gpu_top_blob_int8_scale, &top_blob_int8_scale, sizeof(float), cudaMemcpyHostToDevice));
    }

    return 0;
}

int ConvolutionDepthWise_cuda::load_model(const ModelBin& mb)
{
    int result = ConvolutionDepthWise::load_model(mb);
    if (result < 0)
        return result;

    std::shared_ptr<ncnn::CudaAllocator> cuda_allocator = ncnn::get_current_gpu_allocator();

    gpu_weight_data = CudaMat{weight_data, cuda_allocator};

    if (bias_term)
    {
        gpu_bias_data = CudaMat{bias_data, cuda_allocator};
    }

    if (int8_scale_term)
    {
        gpu_weight_data_int8_scales = CudaMat{weight_data_int8_scales, cuda_allocator};
        gpu_bottom_blob_int8_scales = CudaMat{bottom_blob_int8_scales, cuda_allocator};
        checkCudaErrors(cudaMemcpy(gpu_top_blob_int8_scale, &top_blob_int8_scale, sizeof(float), cudaMemcpyHostToDevice));
    }

    return 0;
}

void ConvolutionDepthWise_cuda::make_padding(const CudaMat& bottom_blob, CudaMat& bottom_blob_bordered, const Option& opt) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    bottom_blob_bordered = bottom_blob;
    if (pad_left > 0 || pad_right > 0 || pad_top > 0 || pad_bottom > 0)
    {
        Option opt_b = opt;
        opt_b.blob_allocator = opt.workspace_allocator;
        copy_make_border(bottom_blob, bottom_blob_bordered, pad_top, pad_bottom, pad_left, pad_right, BORDER_CONSTANT, pad_value, opt_b);
    }
    else if (pad_left == -233 && pad_right == -233 && pad_top == -233 && pad_bottom == -233)
    {
        // tensorflow padding=SAME or onnx padding=SAME_UPPER
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;
        if (wpad > 0 || hpad > 0)
        {
            Option opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            copy_make_border(bottom_blob, bottom_blob_bordered, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, BORDER_CONSTANT, pad_value, opt_b);
        }
    }
    else if (pad_left == -234 && pad_right == -234 && pad_top == -234 && pad_bottom == -234)
    {
        // onnx padding=SAME_LOWER
        int wpad = kernel_extent_w + (w - 1) / stride_w * stride_w - w;
        int hpad = kernel_extent_h + (h - 1) / stride_h * stride_h - h;
        if (wpad > 0 || hpad > 0)
        {
            Option opt_b = opt;
            opt_b.blob_allocator = opt.workspace_allocator;
            copy_make_border(bottom_blob, bottom_blob_bordered, hpad - hpad / 2, hpad / 2, wpad - wpad / 2, wpad / 2, BORDER_CONSTANT, pad_value, opt_b);
        }
    }
}

int ConvolutionDepthWise_cuda::create_pipeline(const Option& opt)
{
    const int elemsize =  weight_data.elemsize;
    ConvolutionDepthWise::create_pipeline(opt);

    if (opt.use_int8_inference && elemsize == (size_t)4u && int8_scale_term)
    {
        std::shared_ptr<ncnn::CudaAllocator> cuda_allocator = ncnn::get_current_gpu_allocator();
        gpu_weight_data = CudaMat{weight_data, cuda_allocator};
    }

    return 0;
}

int ConvolutionDepthWise_cuda::forward(const CudaMat& bottom_blob, CudaMat& top_blob, const Option& opt) const
{
#if LOG_LAYERS
    LOGL("ConvolutionDepthWise_cuda forward");
#endif
    if (opt.use_int8_inference && weight_data.elemsize == (size_t)1u)
    {
        return forward_int8(bottom_blob, top_blob, opt);
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

    if (channels % group != 0 || num_output % group != 0)
    {
        // reject invalid group
        return -100;
    }

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    CudaMat bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;

    int outw = (w - kernel_extent_w) / stride_w + 1;
    int outh = (h - kernel_extent_h) / stride_h + 1;

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w * dilation_h - kernel_w * dilation_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }

    std::shared_ptr<ncnn::CudaAllocator> cuda_allocator = ncnn::get_current_gpu_allocator();

    // float32
    // float32
    top_blob.create(outw, outh, num_output, elemsize, cuda_allocator);
    if (top_blob.empty())
        return -100;

    return convolutiondepthwise_cuda_forward(bottom_blob_bordered, top_blob, ConvolutionDepthWise_info(*this, _space_ofs));
}


int ConvolutionDepthWise_cuda::forward_int8(const CudaMat& bottom_blob, CudaMat& top_blob, const Option& opt) const
{
#if LOG_LAYERS
    LOGL("ConvolutionDepthWise_cuda forward_int8");
#endif
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    size_t elemsize = bottom_blob.elemsize;

    if (channels % group != 0 || num_output % group != 0)
    {
        // reject invalid group
        return -100;
    }

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    std::shared_ptr<ncnn::CudaAllocator> cuda_allocator = ncnn::get_current_gpu_allocator();

    CudaMat bottom_blob_unbordered = bottom_blob;
    if (elemsize != 1)
    {
        bottom_blob_unbordered.create(w, h, channels, (size_t)1u, cuda_allocator);
        if (bottom_blob_unbordered.empty())
            return -100;

        const int channels_g = channels / group;

        // quantize, scale and round to nearest
        for (int g = 0; g < group; g++)
        {
            Option opt_g = opt;
            opt_g.num_threads = 1;
            opt_g.blob_cuda_allocator = bottom_blob_unbordered.allocator;

            const CudaMat bottom_blob_g = bottom_blob.channel_range(channels_g * g, channels_g);
            CudaMat bottom_blob_int8_g = bottom_blob_unbordered.channel_range(channels_g * g, channels_g);

            quantize_float32_to_int8(bottom_blob_g, bottom_blob_int8_g, bottom_blob_int8_scales[g], opt_g);
        }
    }

    CudaMat bottom_blob_bordered;
    make_padding(bottom_blob_unbordered, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    w = bottom_blob_bordered.w;
    h = bottom_blob_bordered.h;

    int outw = (w - kernel_extent_w) / stride_w + 1;
    int outh = (h - kernel_extent_h) / stride_h + 1;

    const int maxk = kernel_w * kernel_h;

    // kernel offsets
    std::vector<int> _space_ofs(maxk);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w * dilation_h - kernel_w * dilation_w;
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                space_ofs[p1] = p2;
                p1++;
                p2 += dilation_w;
            }
            p2 += gap;
        }
    }

    // int8
    size_t out_elemsize = use_int8_requantize ? 1u : 4u;

    top_blob.create(outw, outh, num_output, out_elemsize, cuda_allocator);
    if (top_blob.empty())
        return -100;

    return convolutiondepthwise_cuda_forward_int8(bottom_blob_bordered, top_blob, ConvolutionDepthWise_info(*this, _space_ofs));
}




} // namespace ncnn
