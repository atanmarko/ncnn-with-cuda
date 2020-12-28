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


#include "crop_cuda.h"

#include <math.h>
#include <vector>

namespace ncnn {

int crop_cuda_forward(const CudaMat& bottom_blob, CudaMat& top_blob, const Crop_cuda::Crop_info crop_info);

Crop_cuda::Crop_cuda()
{
    support_cuda = true;
}

int Crop_cuda::load_param(const ParamDict& pd)
{
    return Crop::load_param(pd);
}

int Crop_cuda::forward(const CudaMat& bottom_blob, CudaMat& top_blob, const Option&) const
{
#if LOG_LAYERS
    LOGL("Crop_cuda forward");
#endif
    int _woffset, _hoffset, _coffset;
    int _outw{-1}, _outh{-1}, _outc;
    resolve_crop_roi(bottom_blob.shape(), _woffset, _hoffset, _coffset, _outw, _outh, _outc);

    return crop_cuda_forward(bottom_blob, top_blob, Crop_info{_woffset, _hoffset, _coffset, _outw, _outh, _outc});
}

int Crop_cuda::forward(const std::vector<CudaMat>& bottom_blobs, std::vector<CudaMat>& top_blobs, const Option&) const
{
#if LOG_LAYERS
    LOGL("Crop_cuda forward 2");
#endif
    const CudaMat& bottom_blob = bottom_blobs[0];
    const CudaMat& reference_blob = bottom_blobs[1];
    CudaMat& top_blob = top_blobs[0];

    int _woffset{0}, _hoffset{0}, _coffset{-1};
    int _outw{-1}, _outh{-1}, _outc{0};
    if (woffset == -233)
    {
        std::shared_ptr<int> reference_blob_data = reference_blob.copy_gpu_data<int>();
        resolve_crop_roi(bottom_blob.shape(), (const int*)reference_blob_data.get(), _woffset, _hoffset, _coffset, _outw, _outh, _outc);
    }
    else
    {
        resolve_crop_roi(bottom_blob.shape(), reference_blob.shape(), _woffset, _hoffset, _coffset, _outw, _outh, _outc);
    }

    return crop_cuda_forward(bottom_blob, top_blob, Crop_info{_woffset, _hoffset, _coffset, _outw, _outh, _outc});

    return 0;
}




void Crop_cuda::resolve_crop_roi(const CudaMat& bottom_blob, int& _woffset, int& _hoffset, int& _coffset, int& _outw, int& _outh, int& _outc) const
{
    int w = bottom_blob.w;
    int h = bottom_blob.h;
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;

    bool numpy_style_slice = !starts.empty() && !ends.empty();
    if (numpy_style_slice)
    {
        _woffset = 0;
        _hoffset = 0;
        _coffset = 0;
        _outw = w;
        _outh = h;
        _outc = channels;

        const int* starts_ptr = starts;
        const int* ends_ptr = ends;
        const int* axes_ptr = axes;

        int _axes[3] = {0, 1, 2};
        int num_axis = axes.w;
        if (num_axis == 0)
        {
            num_axis = dims;
        }
        else
        {
            for (int i = 0; i < num_axis; i++)
            {
                int axis = axes_ptr[i];
                if (axis < 0)
                    axis = dims + axis;
                _axes[i] = axis;
            }
        }

        for (int i = 0; i < num_axis; i++)
        {
            int axis = _axes[i];
            int start = starts_ptr[i];
            int end = ends_ptr[i];

            if (dims == 1) // axis == 0
            {
                if (start == -233) start = 0;
                if (end == -233) end = w;
                _woffset = start >= 0 ? start : w + start;
                _outw = std::min(w, end > 0 ? end : w + end) - _woffset;
            }
            if (dims == 2)
            {
                if (axis == 0)
                {
                    if (start == -233) start = 0;
                    if (end == -233) end = h;
                    _hoffset = start >= 0 ? start : h + start;
                    _outh = std::min(h, end > 0 ? end : h + end) - _hoffset;
                }
                if (axis == 1)
                {
                    if (start == -233) start = 0;
                    if (end == -233) end = w;
                    _woffset = start >= 0 ? start : w + start;
                    _outw = std::min(w, end > 0 ? end : w + end) - _woffset;
                }
            }
            if (dims == 3)
            {
                if (axis == 0)
                {
                    if (start == -233) start = 0;
                    if (end == -233) end = channels;
                    _coffset = start >= 0 ? start : channels + start;
                    _outc = std::min(channels, end > 0 ? end : channels + end) - _coffset;
                }
                if (axis == 1)
                {
                    if (start == -233) start = 0;
                    if (end == -233) end = h;
                    _hoffset = start >= 0 ? start : h + start;
                    _outh = std::min(h, end > 0 ? end : h + end) - _hoffset;
                }
                if (axis == 2)
                {
                    if (start == -233) start = 0;
                    if (end == -233) end = w;
                    _woffset = start >= 0 ? start : w + start;
                    _outw = std::min(w, end > 0 ? end : w + end) - _woffset;
                }
            }
        }
    }
    else
    {
        _woffset = woffset;
        _hoffset = hoffset;
        _coffset = coffset;
        _outw = w;
        _outh = h;
        _outc = channels;

        if (dims == 1)
        {
            _outw = w - woffset - woffset2;
            if (outw != -233)
                _outw = std::min(outw, _outw);
        }
        if (dims == 2)
        {
            if (hoffset == -233)
            {
                _woffset = 0;
                _hoffset = woffset;

                _outw = w;
                _outh = h - woffset - woffset2;
                if (outw != -233)
                    _outh = std::min(outw, _outh);
            }
            else
            {
                _outw = w - woffset - woffset2;
                if (outw != -233)
                    _outw = std::min(outw, _outw);

                _outh = h - hoffset - hoffset2;
                if (outh != -233)
                    _outh = std::min(outh, _outh);
            }
        }
        if (dims == 3)
        {
            if (hoffset == -233 && coffset == -233)
            {
                _woffset = 0;
                _hoffset = 0;

                _outw = w;
                _outh = h;
                _outc = channels - woffset - woffset2;
                if (outw != -233)
                    _outc = std::min(outw, _outc);
            }
            else if (coffset == -233)
            {
                _woffset = 0;

                _outw = w;
                _outh = h - woffset - woffset2;
                if (outw != -233)
                    _outh = std::min(outw, _outh);

                _outc = channels - hoffset - hoffset2;
                if (outh != -233)
                    _outc = std::min(outh, _outc);
            }
            else
            {
                _outw = w - woffset - woffset2;
                if (outw != -233)
                    _outw = std::min(outw, _outw);

                _outh = h - hoffset - hoffset2;
                if (outh != -233)
                    _outh = std::min(outh, _outh);

                _outc = channels - coffset - coffset2;
                if (outc != -233)
                    _outc = std::min(outc, _outc);
            }
        }
    }
}

void Crop_cuda::resolve_crop_roi(const CudaMat& bottom_blob, const CudaMat& reference_blob, int& _woffset, int& _hoffset, int& _coffset, int& _outw, int& _outh, int& _outc) const
{
    int channels = bottom_blob.c;
    int dims = bottom_blob.dims;

    int ref_w = reference_blob.w;
    int ref_h = reference_blob.h;
    int ref_channels = reference_blob.c;
    int ref_dims = reference_blob.dims;

    if (dims == 1)
    {
        _woffset = woffset;
        _outw = ref_w;
    }
    if (dims == 2)
    {
        _woffset = woffset;
        _hoffset = hoffset;
        _outw = ref_w;
        _outh = ref_h;
    }
    if (dims == 3)
    {
        _woffset = woffset;
        _hoffset = hoffset;
        _coffset = coffset;
        _outw = ref_w;
        _outh = ref_h;
        _outc = ref_dims == 3 ? ref_channels : channels;
    }
}

void Crop_cuda::resolve_crop_roi(const CudaMat& bottom_blob, const int* param_data, int& _woffset, int& _hoffset, int& _coffset, int& _outw, int& _outh, int& _outc) const
{
    int dims = bottom_blob.dims;

    if (dims == 1)
    {
        _woffset = param_data[0];
        _outw = param_data[3];
    }
    if (dims == 2)
    {
        _woffset = param_data[0];
        _hoffset = param_data[1];
        _outw = param_data[3];
        _outh = param_data[4];
    }
    if (dims == 3)
    {
        _woffset = param_data[0];
        _hoffset = param_data[1];
        _coffset = param_data[2];
        _outw = param_data[3];
        _outh = param_data[4];
        _outc = param_data[5];
    }
}



} // namespace ncnn
