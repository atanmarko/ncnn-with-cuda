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


#ifndef LAYER_CROP_CUDA_H
#define LAYER_CROP_CUDA_H

#include "crop.h"

namespace ncnn {

class Crop_cuda : virtual public Crop
{
public:
    struct Crop_info
    {
        Crop_info(int _woffset, int _hoffset, int _coffset, int _outw, int _outh, int _outc)
            : woffset(_woffset), hoffset(_hoffset), coffset(_coffset), outw(_outw), outh(_outh), outc(_outc)
        {
        }

        int woffset;
        int hoffset;
        int coffset;
        int outw;
        int outh;
        int outc;
    };

    Crop_cuda();

    virtual int load_param(const ParamDict& pd);

    using Crop::forward;

    virtual int forward(const CudaMat& bottom_blob, CudaMat& top_blob, const Option& opt) const;

    virtual int forward(const std::vector<CudaMat>& bottom_blobs, std::vector<CudaMat>& top_blobs, const Option& opt) const;

protected:
    void resolve_crop_roi(const CudaMat& bottom_blob, int& woffset, int& hoffset, int& coffset, int& outw, int& outh, int& outc) const;
    void resolve_crop_roi(const CudaMat& bottom_blob, const CudaMat& reference_blob, int& woffset, int& hoffset, int& coffset, int& outw, int& outh, int& outc) const;
    void resolve_crop_roi(const CudaMat& bottom_blob, const int* param_data, int& woffset, int& hoffset, int& coffset, int& outw, int& outh, int& outc) const;

};

} // namespace ncnn

#endif // LAYER_CROP_CUDA_H
