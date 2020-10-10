//
// Copyright (C) 2020 TANCOM SOFTWARE SOLUTIONS LLC. All rights reserved.
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
//

#ifndef LAYER_BATCHNORM_CUDA_H
#define LAYER_BATCHNORM_CUDA_H

#include "batchnorm.h"

namespace ncnn {

class BatchNorm_cuda : virtual public BatchNorm
{
public:
    BatchNorm_cuda();

    virtual int create_pipeline(const Option& /*opt*/);
    virtual int destroy_pipeline(const Option& /*opt*/);

    virtual int load_model(const CudaModelBinFromMatArray& mb);

    virtual int forward_inplace(CudaMat& bottom_top_blob, const Option& opt) const;

public:
    // model
    CudaMat slope_data_gpu;
    CudaMat mean_data_gpu;
    CudaMat var_data_gpu;
    CudaMat bias_data_gpu;

    mutable CudaMat a_data_gpu;
    mutable CudaMat b_data_gpu;

private:
    std::shared_ptr<CudaAllocator> _allocator;

};

} // namespace ncnn

#endif // LAYER_BATCHNORM_CUDA_H
