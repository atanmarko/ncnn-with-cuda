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


#ifndef LAYER_POOLING_CUDA_H
#define LAYER_POOLING_CUDA_H

#include "pooling.h"

namespace ncnn {

class Pooling_cuda : virtual public Pooling
{
public:
    Pooling_cuda();

    ~Pooling_cuda();

    using Pooling::forward;

    virtual int forward(const CudaMat& bottom_blob, CudaMat& top_blob, const Option& opt) const;

    struct Pooling_info {
        Pooling_info(const Pooling_cuda& poolingCuda, const std::vector<int>& _space_ofs, const Option& opt)
            : pooling_type{poolingCuda.pooling_type},
              kernel_w{poolingCuda.kernel_w},
              kernel_h{poolingCuda.kernel_h},
              stride_w{poolingCuda.stride_w},
              stride_h{poolingCuda.stride_h},
              pad_left{poolingCuda.pad_left},
              pad_right{poolingCuda.pad_right},
              pad_top{poolingCuda.pad_top},
              pad_bottom{poolingCuda.pad_bottom},
              global_pooling{poolingCuda.global_pooling},
              pad_mode{poolingCuda.pad_mode},
              avgpool_count_include_pad{poolingCuda.avgpool_count_include_pad},
              pooling_lock{poolingCuda.pooling_lock}

        {
            maxk = kernel_w * kernel_h;

            if (!_space_ofs.empty())
            {
                auto deleter = [opt](int* pointer) {
                    opt.blob_cuda_allocator->fastFree(pointer);
                };
                gpu_space_ofs = std::shared_ptr<int>(static_cast<int*>(opt.blob_cuda_allocator->fastMalloc(_space_ofs.size() * sizeof(int))), deleter);
                checkCudaErrors(cudaMemcpy(gpu_space_ofs.get(), _space_ofs.data(), _space_ofs.size() * sizeof(int), cudaMemcpyHostToDevice));
            }

        }

        int pooling_type;
        int kernel_w;
        int kernel_h;
        int stride_w;
        int stride_h;
        int pad_left;
        int pad_right;
        int pad_top;
        int pad_bottom;
        int global_pooling;
        int pad_mode; // 0=full 1=valid 2=SAME_UPPER 3=SAME_LOWER
        int avgpool_count_include_pad;
        int* pooling_lock;

        std::shared_ptr<int> gpu_space_ofs;

        int maxk;
        int wtailpad{0};
        int htailpad{0};

    };

    int* pooling_lock;

protected:
    void make_padding(const CudaMat& bottom_blob, CudaMat& bottom_blob_bordered, const Option& opt) const;



};

} // namespace ncnn

#endif // LAYER_POOLING_CUDA_H
