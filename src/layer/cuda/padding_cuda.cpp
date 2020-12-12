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


#include "padding_cuda.h"

#include <math.h>
#include <vector>

namespace ncnn {

int copy_make_border_image(const CudaMat& src, CudaMat& dst, int top, int left, int type, PaddingValue value, PaddingVariableType padding_type);
int copy_make_border_image_3d(const CudaMat& src, CudaMat& dst, int front, int top, int left, int type,
                              PaddingValue value, PaddingVariableType padding_type,
                              void *gpu_per_channel_padding_data, int per_channel_pad_data_size);

Padding_cuda::Padding_cuda()
{
    support_cuda = true;
}



Padding_cuda::~Padding_cuda()
{
    std::shared_ptr<ncnn::CudaAllocator> cuda_allocator = ncnn::get_current_gpu_allocator();
    cuda_allocator->fastFree(gpu_per_channel_pad_data_char);
    cuda_allocator->fastFree(gpu_per_channel_pad_data_unsigned_short);
    cuda_allocator->fastFree(gpu_per_channel_pad_data_unsigned_short_use_fp16);
    cuda_allocator->fastFree(gpu_per_channel_pad_data_float);

}

int Padding_cuda::load_param(const ParamDict& pd)
{
    return Padding::load_param(pd);
}

int Padding_cuda::load_model(const CudaModelBinFromMatArray& pd)
{
    Padding::load_model(static_cast<const ModelBinFromMatArray>(pd));

    std::shared_ptr<ncnn::CudaAllocator> cuda_allocator = ncnn::get_current_gpu_allocator();
    gpu_per_channel_pad_data = CudaMat{per_channel_pad_data, cuda_allocator};

    const int number_of_elements = per_channel_pad_data.w;

    std::unique_ptr<char[]> padding_char = std::make_unique<char[]>(number_of_elements);
    std::unique_ptr<unsigned short[]> padding_unsigned_short = std::make_unique<unsigned short[]>(number_of_elements);
    std::unique_ptr<unsigned short[]> padding_unsigned_short_use_fp16 = std::make_unique<unsigned short[]>(number_of_elements);
    std::unique_ptr<float[]> padding_float = std::make_unique<float[]>(number_of_elements);


    for (int i=0; i<number_of_elements; ++i) {
        float pad_value = per_channel_pad_data[i];
        padding_char[i] = static_cast<signed char>(pad_value);
        padding_unsigned_short[i] =float32_to_bfloat16(pad_value);
        padding_unsigned_short_use_fp16[i] = float32_to_float16(pad_value);
        padding_float[i] = pad_value;
    }

    gpu_per_channel_pad_data_char = static_cast<char*>(cuda_allocator->fastMalloc(sizeof(signed char) * number_of_elements));
    gpu_per_channel_pad_data_unsigned_short = static_cast<unsigned short*>(cuda_allocator->fastMalloc(sizeof(unsigned short) * number_of_elements));
    gpu_per_channel_pad_data_unsigned_short_use_fp16 = static_cast<unsigned short*>(cuda_allocator->fastMalloc(sizeof(unsigned short) *number_of_elements));
    gpu_per_channel_pad_data_float = static_cast<float*>(cuda_allocator->fastMalloc(sizeof(float) * number_of_elements));


    checkCudaErrors(cudaMemcpy(gpu_per_channel_pad_data_char, padding_char.get(), sizeof(signed char) * number_of_elements, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(gpu_per_channel_pad_data_unsigned_short, padding_unsigned_short.get(), sizeof(unsigned short) * number_of_elements, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(gpu_per_channel_pad_data_unsigned_short_use_fp16, padding_unsigned_short_use_fp16.get(), sizeof(unsigned short) * number_of_elements, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(gpu_per_channel_pad_data_float, padding_float.get(), sizeof(float) * number_of_elements, cudaMemcpyHostToDevice));

    return 0;
}


int Padding_cuda::forward(const CudaMat& bottom_blob, CudaMat& top_blob, const Option& opt) const
{
    if (top == 0 && bottom == 0 && left == 0 && right == 0 && front == 0 && behind == 0)
    {
        top_blob = bottom_blob;
        return 0;
    }

    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;
    size_t elemsize = bottom_blob.elemsize;

    std::shared_ptr<ncnn::CudaAllocator> cuda_allocator = ncnn::get_current_gpu_allocator();

    int outw = w + left + right;
    int outh = h + top + bottom;
    int outc = channels + front + behind;

    int top_value = top;

    if (dims == 1 || dims == 2)
    {
        if (dims == 1)
        {
            top_value = 0;
            top_blob.create(outw, elemsize, cuda_allocator);
            if (top_blob.empty())
                return -100;
        }

        if (dims == 2)
        {
            top_blob.create(outw, outh, elemsize, cuda_allocator);
            if (top_blob.empty())
                return -100;
        }

        //dims 1 and dims 2
        PaddingValue param_value{0};
        if (elemsize == 1) {
            param_value.c = static_cast<signed char>(value);
            copy_make_border_image(bottom_blob, top_blob, top_value, left, type, param_value, PaddingVariableType::type_char);
        }
        if (elemsize == 2)
        {
            param_value.sh = opt.use_fp16_storage ? float32_to_float16(value) : float32_to_bfloat16(value);
            copy_make_border_image(bottom_blob, top_blob, top_value, left, type, param_value, PaddingVariableType::type_unsigned_short);
        }
        if (elemsize == 4)
        {
            param_value.fl = value;
            copy_make_border_image(bottom_blob, top_blob, top_value, left, type, param_value, PaddingVariableType::type_float);
        }

    }
    else if (dims == 3)
    {
        top_blob.create(outw, outh, outc, elemsize, cuda_allocator);
        if (top_blob.empty())
            return -100;

        PaddingValue param_value{0};
        if (elemsize == 1) {
            param_value.c = static_cast<signed char>(value);
            copy_make_border_image_3d(bottom_blob, top_blob, front, top_value, left, type, param_value, PaddingVariableType::type_char,
                                      static_cast<void*>(gpu_per_channel_pad_data_char), per_channel_pad_data_size);
        }
        if (elemsize == 2)
        {
            param_value.sh = opt.use_fp16_storage ? float32_to_float16(value) : float32_to_bfloat16(value);
            copy_make_border_image_3d(bottom_blob, top_blob, front, top_value, left, type, param_value, PaddingVariableType::type_unsigned_short,
                                      opt.use_fp16_storage ? static_cast<void*>(gpu_per_channel_pad_data_unsigned_short_use_fp16):
                                      static_cast<void*>(gpu_per_channel_pad_data_unsigned_short), per_channel_pad_data_size);
        }
        if (elemsize == 4)
        {
            param_value.fl = value;
            copy_make_border_image_3d(bottom_blob, top_blob, front, top_value, left, type, param_value, PaddingVariableType::type_float,
                                      static_cast<void*>(gpu_per_channel_pad_data_float), per_channel_pad_data_size);
        }


    }


    return 0;

}

int Padding_cuda::forward(const std::vector<CudaMat>&, std::vector<CudaMat>&, const Option&) const
{
    std::cout << "Padding_cuda::forward 2" << std::endl;
    return 0;
}


} // namespace ncnn
