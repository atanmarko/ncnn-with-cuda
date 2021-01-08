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
#include "convolution_cuda.h"

#include <chrono>

namespace ncnn {

int convolution_cuda_forward(const CudaMat& bottom_blob, CudaMat& top_blob, const Convolution_cuda::Convolution_info& info);
int convolution_cuda_forward_03(const CudaMat& bottom_blob, CudaMat& top_blob, const Convolution_cuda::Convolution_info& info);
int convolution_cuda_forward_int8(const CudaMat& bottom_blob, CudaMat& top_blob, const Convolution_cuda::Convolution_info& info);

Convolution_cuda::Convolution_cuda()
{
    support_cuda = true;
    _cuda_allocator = ncnn::get_current_gpu_allocator();
    gpu_bottom_blob_int8_scale = static_cast<float *>(_cuda_allocator->fastMalloc(sizeof(float)));
    gpu_top_blob_int8_scale = static_cast<float *>(_cuda_allocator->fastMalloc(sizeof(float)));
}

Convolution_cuda::~Convolution_cuda()
{
    _cuda_allocator->fastFree(gpu_bottom_blob_int8_scale);
    _cuda_allocator->fastFree(gpu_top_blob_int8_scale);
}

int Convolution_cuda::load_param(const ParamDict& pd)
{
    Convolution::load_param(pd);

    gpu_activation_params = CudaMat{activation_params, _cuda_allocator};

    return 0;
}

int Convolution_cuda::load_model(const CudaModelBinFromMatArray& mb)
{
    int result = Convolution::load_model(static_cast<ModelBinFromMatArray>(mb));
    if (result < 0)
        return result;

    gpu_weight_data = CudaMat{weight_data, _cuda_allocator};

    if (bias_term)
    {
        gpu_bias_data = CudaMat{bias_data, _cuda_allocator};
    }

    if (int8_scale_term)
    {
        gpu_weight_data_int8_scales = CudaMat{weight_data_int8_scales, _cuda_allocator};
        checkCudaErrors(cudaMemcpy(gpu_bottom_blob_int8_scale, &bottom_blob_int8_scale, sizeof(float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(gpu_top_blob_int8_scale, &top_blob_int8_scale, sizeof(float), cudaMemcpyHostToDevice));
    }

    return 0;
}

int Convolution_cuda::load_model(const ModelBin& mb)
{
    int result = Convolution::load_model(mb);
    if (result < 0)
        return result;

    gpu_weight_data = CudaMat{weight_data, _cuda_allocator};

    if (bias_term)
    {
        gpu_bias_data = CudaMat{bias_data, _cuda_allocator};
    }

    if (int8_scale_term)
    {
        gpu_weight_data_int8_scales = CudaMat{weight_data_int8_scales, _cuda_allocator};
        checkCudaErrors(cudaMemcpy(gpu_bottom_blob_int8_scale, &bottom_blob_int8_scale, sizeof(float), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(gpu_top_blob_int8_scale, &top_blob_int8_scale, sizeof(float), cudaMemcpyHostToDevice));
    }

    return 0;
}

void Convolution_cuda::make_padding(const CudaMat& bottom_blob, CudaMat& bottom_blob_bordered, const Option& opt) const
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
        opt_b.blob_cuda_allocator = opt.workspace_cuda_allocator;
        copy_make_border(bottom_blob, bottom_blob_bordered, pad_top, pad_bottom, pad_left, pad_right,
                         BORDER_CONSTANT, pad_value, opt_b);
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
            opt_b.blob_cuda_allocator = opt.workspace_cuda_allocator;
            copy_make_border(bottom_blob, bottom_blob_bordered, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2,
                             BORDER_CONSTANT, pad_value, opt_b);
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
            opt_b.blob_cuda_allocator = opt.workspace_cuda_allocator;
            copy_make_border(bottom_blob, bottom_blob_bordered, hpad - hpad / 2, hpad / 2, wpad - wpad / 2, wpad / 2,
                             BORDER_CONSTANT, pad_value, opt_b);
        }
    }
}

int Convolution_cuda::create_pipeline(const Option& opt)
{
    const int elemsize =  weight_data.elemsize;
    Convolution::create_pipeline(opt);

    if (opt.use_int8_inference && elemsize == (size_t)4u && int8_scale_term)
    {
        gpu_weight_data = CudaMat{weight_data, opt.blob_cuda_allocator};
    }

    return 0;
}

int Convolution_cuda::forward(const CudaMat& bottom_blob, CudaMat& top_blob, const Option& opt) const
{
#if LOG_LAYERS
    LOGL("Convolution_cuda forward");
#endif
    if (opt.use_int8_inference && weight_data.elemsize == (size_t)1u)
    {
        return forward_int8(bottom_blob, top_blob, opt);
    }

    // flattened blob, implement as InnerProduct
    if (bottom_blob.dims == 1 && kernel_w == 1 && kernel_h == 1)
    {
        int num_input = weight_data_size / num_output;
        if (bottom_blob.w * bottom_blob.elempack == num_input)
        {
            // call InnerProduct
            ncnn::Layer* op = ncnn::create_layer(ncnn::LayerType::InnerProduct);

            // set param
            ncnn::ParamDict pd;
            pd.set(0, num_output);
            pd.set(1, bias_term);
            pd.set(2, weight_data_size);
            pd.set(8, int8_scale_term);
            pd.set(9, activation_type);
            pd.set(10, activation_params);

            op->load_param(pd);

            // set weights
            ncnn::Mat weights[4];
            weights[0] = weight_data;
            weights[1] = bias_data;

            if (int8_scale_term)
            {
                weights[2] = weight_data_int8_scales;
                weights[3] = Mat(1, (size_t)4u, (void*)& bottom_blob_int8_scale);
            }

            op->load_model(CudaModelBinFromMatArray(weights));

            op->create_pipeline(opt);

            // forward
            op->forward(bottom_blob, top_blob, opt);

            op->destroy_pipeline(opt);

            delete op;

            return 0;
        }
    }


    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    CudaMat bottom_blob_bordered;
    make_padding(bottom_blob, bottom_blob_bordered, opt);
    if (bottom_blob_bordered.empty())
        return -100;

    int w = bottom_blob_bordered.w;
    int h = bottom_blob_bordered.h;
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

    // float32
    top_blob.create(outw, outh, num_output, bottom_blob.elemsize, opt.blob_cuda_allocator);
    if (top_blob.empty())
        return -100;

    int *gpu_space_ofs = static_cast<int*>(opt.blob_cuda_allocator->fastMalloc(_space_ofs.size() * sizeof(int)));
    checkCudaErrors(cudaMemcpy(gpu_space_ofs, _space_ofs.data(), _space_ofs.size() * sizeof(int), cudaMemcpyHostToDevice));

//    int result = convolution_cuda_forward_03(bottom_blob_bordered, top_blob, Convolution_info(*this, gpu_space_ofs));
    int result = convolution_cuda_forward(bottom_blob_bordered, top_blob, Convolution_info(*this, gpu_space_ofs));

    opt.blob_cuda_allocator->fastFree(gpu_space_ofs);

    return result;
}


int Convolution_cuda::forward_int8(const CudaMat& bottom_blob, CudaMat& top_blob, const Option& opt) const
{
#if LOG_LAYERS
    LOGL("Convolution_cuda forward_int8");
#endif
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    size_t elemsize = bottom_blob.elemsize;

    const int kernel_extent_w = dilation_w * (kernel_w - 1) + 1;
    const int kernel_extent_h = dilation_h * (kernel_h - 1) + 1;

    CudaMat bottom_blob_unbordered = bottom_blob;
    if (elemsize != 1)
    {
        Option opt_g = opt;
        opt_g.blob_allocator = opt.workspace_allocator;
        opt_g.blob_cuda_allocator = opt.workspace_cuda_allocator;

        quantize_float32_to_int8(bottom_blob, bottom_blob_unbordered, bottom_blob_int8_scale, opt_g);
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

    top_blob.create(outw, outh, num_output, out_elemsize, opt.blob_cuda_allocator);
    if (top_blob.empty())
        return -100;

    int *gpu_space_ofs = static_cast<int*>(opt.blob_cuda_allocator->fastMalloc(_space_ofs.size() * sizeof(int)));
    checkCudaErrors(cudaMemcpy(gpu_space_ofs, _space_ofs.data(), _space_ofs.size() * sizeof(int), cudaMemcpyHostToDevice));
    int result = convolution_cuda_forward_int8(bottom_blob_bordered, top_blob, Convolution_info(*this, gpu_space_ofs));
    opt.blob_cuda_allocator->fastFree(gpu_space_ofs);

    return result;
}




} // namespace ncnn
