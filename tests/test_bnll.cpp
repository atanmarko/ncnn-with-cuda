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

#include "layer/bnll.h"
#include "testutil.h"

#if NCNN_CUDA
#include <cuda_profiler_api.h>
#endif


static int test_bnll_cpu(const ncnn::Mat& a)
{

    int channels;
    if (a.dims == 1) channels = a.w;
    if (a.dims == 2) channels = a.h;
    if (a.dims == 3) channels = a.c;

    ncnn::ParamDict pd;
    pd.set(0, channels); // channels

    std::vector<ncnn::Mat> weights(1);
    weights[0] = RandomMat(a.w, a.h, a.c);

    int ret = test_layer<ncnn::BNLL>("BNLL", pd, weights, a);
    if (ret != 0)
    {
        fprintf(stderr, "test_bnll failed a.dims=%d a=(%d %d %d)\n", a.dims, a.w, a.h, a.c);
    }

    return ret;
}


static int test_bnll(const ncnn::Mat& a)
{
    return 0
           || test_bnll_cpu(a)
           ;
}

static int test_bnll_0()
{
    return 0
           || test_bnll(RandomMat(5, 7, 16))
           || test_bnll(RandomMat(3, 5, 13))
           || test_bnll(RandomMat(5, 7, 16))
           || test_bnll(RandomMat(3, 5, 13))
           || test_bnll(RandomMat(5, 7, 16))
           || test_bnll(RandomMat(3, 5, 13))
           || test_bnll(RandomMat(5, 7, 16))
           || test_bnll(RandomMat(3, 5, 13));
}

static int test_bnll_1()
{
    return 0
           || test_bnll(RandomMat(6, 16))
           || test_bnll(RandomMat(7, 15))
           || test_bnll(RandomMat(6, 16))
           || test_bnll(RandomMat(7, 15))
           || test_bnll(RandomMat(6, 16))
           || test_bnll(RandomMat(7, 15))
           || test_bnll(RandomMat(6, 16))
           || test_bnll(RandomMat(7, 15));
}

static int test_bnll_2()
{
    return 0
           || test_bnll(RandomMat(10))
           || test_bnll(RandomMat(127))
           || test_bnll(RandomMat(128))
           || test_bnll(RandomMat(127))
           || test_bnll(RandomMat(128))
           || test_bnll(RandomMat(127))
           || test_bnll(RandomMat(128))
           || test_bnll(RandomMat(127));
}

int main()
{
    SRAND(7767517);

#if NCNN_CUDA
    cudaProfilerStart();
#endif
    int result = 0
           || test_bnll_0()
           || test_bnll_1()
           || test_bnll_2()
        ;
#if NCNN_CUDA
    cudaProfilerStop();
#endif

}
