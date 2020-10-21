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


#ifndef LAYER_PACKING_CUDA_H
#define LAYER_PACKING_CUDA_H

#include "packing.h"

namespace ncnn {

class Packing_cuda : virtual public Packing
{
public:
    Packing_cuda();

    virtual int load_param(const ParamDict& pd);

    virtual int forward(const CudaMat& bottom_blob, CudaMat& top_blob, const Option& opt) const;

    struct packing_options {
        packing_options(const Packing_cuda& pack)
            : out_elempack{pack.out_elempack},
              use_padding{pack.use_padding},
              cast_type_from{pack.cast_type_from},
              cast_type_to{pack.cast_type_to},
              storage_type_from{pack.storage_type_from},
              storage_type_to{pack.storage_type_to}
        {

        }

        int out_elempack;
        int use_padding;
        int cast_type_from;
        int cast_type_to;
        int storage_type_from;
        int storage_type_to;

    };

};

} // namespace ncnn

#endif // LAYER_PACKING_CUDA_H
