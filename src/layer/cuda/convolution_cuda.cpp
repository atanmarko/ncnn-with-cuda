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

int flatten_cuda_forward(const unsigned char* bottom_blob, const ncnn::CudaMatInfo bottom_blob_info,
                         unsigned char* top_blob);

Convolution_cuda::Convolution_cuda()
{
    support_cuda = true;
}

int Convolution_cuda::load_param(const ParamDict& pd)
{
    Convolution::load_param(pd);

    std::shared_ptr<ncnn::CudaAllocator> cuda_allocator = ncnn::get_current_gpu_allocator();
    gpu_activation_params = CudaMat{activation_params, cuda_allocator};

    return 0;
}

int Convolution_cuda::load_model(const ModelBin& mb)
{
    int result = Convolution::load_model(mb);
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
        checkCudaErrors(cudaMemcpy(gpu_bottom_blob_int8_scale, &bottom_blob_int8_scale, sizeof(float), cudaMemcpyHostToDevice));
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
            copy_make_border(bottom_blob, bottom_blob_bordered, hpad - hpad / 2, hpad / 2, wpad - wpad / 2, wpad / 2,
                             BORDER_CONSTANT, pad_value, opt_b);
        }
    }
}

int Convolution_cuda::create_pipeline(const Option& opt)
{
    Convolution::create_pipeline(opt);

    if (opt.use_int8_inference && weight_data.elemsize == (size_t)4u && int8_scale_term)
    {
        std::shared_ptr<ncnn::CudaAllocator> cuda_allocator = ncnn::get_current_gpu_allocator();
        gpu_weight_data = CudaMat{weight_data, cuda_allocator};
    }

    return 0;
}

int Convolution_cuda::forward(const CudaMat& bottom_blob, CudaMat& top_blob, const Option& opt) const
{
    if (opt.use_int8_inference && weight_data.elemsize == (size_t)1u)
    {
        throw std::runtime_error("Not supported");
        return -1;
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
                weights[3] = CudaMat(1, (size_t)4u, (void*)& bottom_blob_int8_scale);
            }

            op->load_model(ModelBinFromMatArray(weights));

            op->create_pipeline(opt);

            // forward
            op->forward(bottom_blob, top_blob, opt);

            op->destroy_pipeline(opt);

            delete op;

            return 0;
        }
    }




    return 0;
}

} // namespace ncnn
