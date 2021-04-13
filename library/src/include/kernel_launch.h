/******************************************************************************
* Copyright (c) 2016 - present Advanced Micro Devices, Inc. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*******************************************************************************/

#ifndef KERNEL_LAUNCH_SINGLE
#define KERNEL_LAUNCH_SINGLE

#define FN_PRFX(X) rocfft_internal_##X
#ifndef __clang__
#include "error.h"
#endif
#include "../../../shared/array_predicate.h"
#include "kargs.h"
#include "rocfft.h"
#include "rocfft_hip.h"
#include "tree_node.h"
#include <iostream>

// FIXME: documentation
struct DeviceCallIn
{
    TreeNode* node;
    void*     bufIn[2];
    void*     bufOut[2];

    hipStream_t     rocfft_stream;
    GridParam       gridParam;
    hipDeviceProp_t deviceProp;
};

// FIXME: documentation
struct DeviceCallOut
{
    int err;
};

/* Naming convention

dfn – device function caller (just a prefix, though actually GPU kernel
function)

sp (dp) – single (double) precision

ip – in-place

op - out-of-place

ci – complex-interleaved (format of input buffer)

ci – complex-interleaved (format of output buffer)

stoc – stockham fft kernel
bcc - block column column

1(2) – one (two) dimension data from kernel viewpoint, but 2D may transform into
1D. e.g  64*128(2D) = 8192(1D)

1024, 64_128 – length of fft on each dimension

*/

void rocfft_internal_mul(const void* data_p, void* back_p);
void rocfft_internal_chirp(const void* data_p, void* back_p);
void rocfft_internal_transpose_var2(const void* data_p, void* back_p);

/*
   data->node->devKernArg : points to the internal length device pointer
   data->node->devKernArg + 1*KERN_ARGS_ARRAY_WIDTH : points to the intenal in
   stride device pointer
   data->node->devKernArg + 2*KERN_ARGS_ARRAY_WIDTH : points to the internal out
   stride device pointer, only used in outof place kernels
*/

/*
    TODO:
        - compress the below code
        - refactor the code to support batched r2c/c2r
 */

// base args for out-of-place includes extra strides for output
#define KERNEL_BASE_ARGS_IP(PRECISION) \
    const PRECISION* __restrict__, const size_t, const size_t*, const size_t*, const size_t
#define KERNEL_BASE_ARGS_OP(PRECISION)                                                        \
    const PRECISION* __restrict__, const size_t, const size_t*, const size_t*, const size_t*, \
        const size_t
#define GET_KERNEL_FUNC(FWD, BACK, PRECISION, BASE_ARGS, ...)         \
    void (*kernel_func)(BASE_ARGS(PRECISION), __VA_ARGS__) = nullptr; \
    if(data->node->inStride[0] == 1 && data->node->outStride[0] == 1) \
    {                                                                 \
        if(data->node->direction == -1)                               \
        {                                                             \
            kernel_func = FWD<PRECISION, SB_UNIT>;                    \
        }                                                             \
        else                                                          \
        {                                                             \
            kernel_func = BACK<PRECISION, SB_UNIT>;                   \
        }                                                             \
    }                                                                 \
    else                                                              \
    {                                                                 \
        if(data->node->direction == -1)                               \
        {                                                             \
            kernel_func = FWD<PRECISION, SB_NONUNIT>;                 \
        }                                                             \
        else                                                          \
        {                                                             \
            kernel_func = BACK<PRECISION, SB_NONUNIT>;                \
        }                                                             \
    }

// SBCC adds large twiddles
#define KERNEL_BASE_ARGS_IP_SBCC(PRECISION)                                                    \
    const PRECISION* __restrict__, const PRECISION* __restrict__, const size_t, const size_t*, \
        const size_t*, const size_t
#define KERNEL_BASE_ARGS_OP_SBCC(PRECISION)                                                    \
    const PRECISION* __restrict__, const PRECISION* __restrict__, const size_t, const size_t*, \
        const size_t*, const size_t*, const size_t
#define GET_KERNEL_FUNC_SBCC(FWD, BACK, PRECISION, BASE_ARGS, ...)      \
    void (*kernel_func)(BASE_ARGS(PRECISION), __VA_ARGS__) = nullptr;   \
    if(data->node->inStride[0] == 1 && data->node->outStride[0] == 1)   \
    {                                                                   \
        if(data->node->direction == -1)                                 \
        {                                                               \
            if(data->node->large1D)                                     \
            {                                                           \
                if(data->node->largeTwdBase == 4)                       \
                    kernel_func = FWD<PRECISION, SB_UNIT, true, 4>;     \
                else if(data->node->largeTwdBase == 5)                  \
                    kernel_func = FWD<PRECISION, SB_UNIT, true, 5>;     \
                else if(data->node->largeTwdBase == 6)                  \
                    kernel_func = FWD<PRECISION, SB_UNIT, true, 6>;     \
                else                                                    \
                    kernel_func = FWD<PRECISION, SB_UNIT, true>;        \
            }                                                           \
            else                                                        \
                kernel_func = FWD<PRECISION, SB_UNIT, false>;           \
        }                                                               \
        else                                                            \
        {                                                               \
            if(data->node->large1D)                                     \
            {                                                           \
                if(data->node->largeTwdBase == 4)                       \
                    kernel_func = BACK<PRECISION, SB_UNIT, true, 4>;    \
                else if(data->node->largeTwdBase == 5)                  \
                    kernel_func = BACK<PRECISION, SB_UNIT, true, 5>;    \
                else if(data->node->largeTwdBase == 6)                  \
                    kernel_func = BACK<PRECISION, SB_UNIT, true, 6>;    \
                else                                                    \
                    kernel_func = BACK<PRECISION, SB_UNIT, true>;       \
            }                                                           \
            else                                                        \
                kernel_func = BACK<PRECISION, SB_UNIT, false>;          \
        }                                                               \
    }                                                                   \
    else                                                                \
    {                                                                   \
        if(data->node->direction == -1)                                 \
        {                                                               \
            if(data->node->large1D)                                     \
            {                                                           \
                if(data->node->largeTwdBase == 4)                       \
                    kernel_func = FWD<PRECISION, SB_NONUNIT, true, 4>;  \
                else if(data->node->largeTwdBase == 5)                  \
                    kernel_func = FWD<PRECISION, SB_NONUNIT, true, 5>;  \
                else if(data->node->largeTwdBase == 6)                  \
                    kernel_func = FWD<PRECISION, SB_NONUNIT, true, 6>;  \
                else                                                    \
                    kernel_func = FWD<PRECISION, SB_NONUNIT, true>;     \
            }                                                           \
            else                                                        \
                kernel_func = FWD<PRECISION, SB_NONUNIT, false>;        \
        }                                                               \
        else                                                            \
        {                                                               \
            if(data->node->large1D)                                     \
            {                                                           \
                if(data->node->largeTwdBase == 4)                       \
                    kernel_func = BACK<PRECISION, SB_NONUNIT, true, 4>; \
                else if(data->node->largeTwdBase == 5)                  \
                    kernel_func = BACK<PRECISION, SB_NONUNIT, true, 5>; \
                else if(data->node->largeTwdBase == 6)                  \
                    kernel_func = BACK<PRECISION, SB_NONUNIT, true, 6>; \
                else                                                    \
                    kernel_func = BACK<PRECISION, SB_NONUNIT, true>;    \
            }                                                           \
            else                                                        \
                kernel_func = BACK<PRECISION, SB_NONUNIT, false>;       \
        }                                                               \
    }

// SBRC has COL_DIM, TRANSPOSE_TYPE template args and is always out-of-place
#define GET_KERNEL_FUNC_SBRC(FWD, BACK, PRECISION, COL_DIM, TRANSPOSE_TYPE, BASE_ARGS, ...) \
    void (*kernel_func)(BASE_ARGS(PRECISION), __VA_ARGS__) = nullptr;                       \
    if(data->node->direction == -1)                                                         \
    {                                                                                       \
        kernel_func = FWD<PRECISION, SB_UNIT, COL_DIM, TRANSPOSE_TYPE>;                     \
    }                                                                                       \
    else                                                                                    \
    {                                                                                       \
        kernel_func = BACK<PRECISION, SB_UNIT, COL_DIM, TRANSPOSE_TYPE>;                    \
    }

#define POWX_SMALL_GENERATOR(FUNCTION_NAME,                                                   \
                             IP_FWD_KERN_NAME,                                                \
                             IP_BACK_KERN_NAME,                                               \
                             OP_FWD_KERN_NAME,                                                \
                             OP_BACK_KERN_NAME,                                               \
                             PRECISION)                                                       \
    void FUNCTION_NAME(const void* data_p, void* back_p)                                      \
    {                                                                                         \
        DeviceCallIn* data          = (DeviceCallIn*)data_p;                                  \
        hipStream_t   rocfft_stream = data->rocfft_stream;                                    \
        if(data->node->placement == rocfft_placement_inplace)                                 \
        {                                                                                     \
            if(array_type_is_interleaved(data->node->inArrayType)                             \
               && array_type_is_interleaved(data->node->outArrayType))                        \
            {                                                                                 \
                GET_KERNEL_FUNC(IP_FWD_KERN_NAME,                                             \
                                IP_BACK_KERN_NAME,                                            \
                                PRECISION,                                                    \
                                KERNEL_BASE_ARGS_IP,                                          \
                                PRECISION* __restrict__);                                     \
                hipLaunchKernelGGL(kernel_func,                                               \
                                   dim3(data->gridParam.b_x),                                 \
                                   dim3(data->gridParam.tpb_x),                               \
                                   0,                                                         \
                                   rocfft_stream,                                             \
                                   (PRECISION*)data->node->twiddles.data(),                   \
                                   data->node->length.size(),                                 \
                                   data->node->devKernArg.data(),                             \
                                   data->node->devKernArg.data() + 1 * KERN_ARGS_ARRAY_WIDTH, \
                                   data->node->batch,                                         \
                                   (PRECISION*)data->bufIn[0]);                               \
            }                                                                                 \
            else if(array_type_is_planar(data->node->inArrayType)                             \
                    && array_type_is_planar(data->node->outArrayType))                        \
            {                                                                                 \
                GET_KERNEL_FUNC(IP_FWD_KERN_NAME,                                             \
                                IP_BACK_KERN_NAME,                                            \
                                PRECISION,                                                    \
                                KERNEL_BASE_ARGS_IP,                                          \
                                real_type_t<PRECISION>* __restrict__,                         \
                                real_type_t<PRECISION>* __restrict__);                        \
                hipLaunchKernelGGL(kernel_func,                                               \
                                   dim3(data->gridParam.b_x),                                 \
                                   dim3(data->gridParam.tpb_x),                               \
                                   0,                                                         \
                                   rocfft_stream,                                             \
                                   (PRECISION*)data->node->twiddles.data(),                   \
                                   data->node->length.size(),                                 \
                                   data->node->devKernArg.data(),                             \
                                   data->node->devKernArg.data() + 1 * KERN_ARGS_ARRAY_WIDTH, \
                                   data->node->batch,                                         \
                                   (real_type_t<PRECISION>*)data->bufIn[0],                   \
                                   (real_type_t<PRECISION>*)data->bufIn[1]);                  \
            }                                                                                 \
        }                                                                                     \
        else /* out of place */                                                               \
        {                                                                                     \
            if(array_type_is_interleaved(data->node->inArrayType)                             \
               && array_type_is_interleaved(data->node->outArrayType))                        \
            {                                                                                 \
                GET_KERNEL_FUNC(OP_FWD_KERN_NAME,                                             \
                                OP_BACK_KERN_NAME,                                            \
                                PRECISION,                                                    \
                                KERNEL_BASE_ARGS_OP,                                          \
                                PRECISION* __restrict__,                                      \
                                PRECISION* __restrict__);                                     \
                hipLaunchKernelGGL(kernel_func,                                               \
                                   dim3(data->gridParam.b_x),                                 \
                                   dim3(data->gridParam.tpb_x),                               \
                                   0,                                                         \
                                   rocfft_stream,                                             \
                                   (PRECISION*)data->node->twiddles.data(),                   \
                                   data->node->length.size(),                                 \
                                   data->node->devKernArg.data(),                             \
                                   data->node->devKernArg.data() + 1 * KERN_ARGS_ARRAY_WIDTH, \
                                   data->node->devKernArg.data() + 2 * KERN_ARGS_ARRAY_WIDTH, \
                                   data->node->batch,                                         \
                                   (PRECISION*)data->bufIn[0],                                \
                                   (PRECISION*)data->bufOut[0]);                              \
            }                                                                                 \
            else if(array_type_is_interleaved(data->node->inArrayType)                        \
                    && array_type_is_planar(data->node->outArrayType))                        \
            {                                                                                 \
                GET_KERNEL_FUNC(OP_FWD_KERN_NAME,                                             \
                                OP_BACK_KERN_NAME,                                            \
                                PRECISION,                                                    \
                                KERNEL_BASE_ARGS_OP,                                          \
                                PRECISION* __restrict__,                                      \
                                real_type_t<PRECISION>* __restrict__,                         \
                                real_type_t<PRECISION>* __restrict__);                        \
                hipLaunchKernelGGL(kernel_func,                                               \
                                   dim3(data->gridParam.b_x),                                 \
                                   dim3(data->gridParam.tpb_x),                               \
                                   0,                                                         \
                                   rocfft_stream,                                             \
                                   (PRECISION*)data->node->twiddles.data(),                   \
                                   data->node->length.size(),                                 \
                                   data->node->devKernArg.data(),                             \
                                   data->node->devKernArg.data() + 1 * KERN_ARGS_ARRAY_WIDTH, \
                                   data->node->devKernArg.data() + 2 * KERN_ARGS_ARRAY_WIDTH, \
                                   data->node->batch,                                         \
                                   (PRECISION*)data->bufIn[0],                                \
                                   (real_type_t<PRECISION>*)data->bufOut[0],                  \
                                   (real_type_t<PRECISION>*)data->bufOut[1]);                 \
            }                                                                                 \
            else if(array_type_is_planar(data->node->inArrayType)                             \
                    && array_type_is_interleaved(data->node->outArrayType))                   \
            {                                                                                 \
                GET_KERNEL_FUNC(OP_FWD_KERN_NAME,                                             \
                                OP_BACK_KERN_NAME,                                            \
                                PRECISION,                                                    \
                                KERNEL_BASE_ARGS_OP,                                          \
                                real_type_t<PRECISION>* __restrict__,                         \
                                real_type_t<PRECISION>* __restrict__,                         \
                                PRECISION* __restrict__);                                     \
                hipLaunchKernelGGL(kernel_func,                                               \
                                   dim3(data->gridParam.b_x),                                 \
                                   dim3(data->gridParam.tpb_x),                               \
                                   0,                                                         \
                                   rocfft_stream,                                             \
                                   (PRECISION*)data->node->twiddles.data(),                   \
                                   data->node->length.size(),                                 \
                                   data->node->devKernArg.data(),                             \
                                   data->node->devKernArg.data() + 1 * KERN_ARGS_ARRAY_WIDTH, \
                                   data->node->devKernArg.data() + 2 * KERN_ARGS_ARRAY_WIDTH, \
                                   data->node->batch,                                         \
                                   (real_type_t<PRECISION>*)data->bufIn[0],                   \
                                   (real_type_t<PRECISION>*)data->bufIn[1],                   \
                                   (PRECISION*)data->bufOut[0]);                              \
            }                                                                                 \
            else if(array_type_is_planar(data->node->inArrayType)                             \
                    && array_type_is_planar(data->node->outArrayType))                        \
            {                                                                                 \
                GET_KERNEL_FUNC(OP_FWD_KERN_NAME,                                             \
                                OP_BACK_KERN_NAME,                                            \
                                PRECISION,                                                    \
                                KERNEL_BASE_ARGS_OP,                                          \
                                real_type_t<PRECISION>* __restrict__,                         \
                                real_type_t<PRECISION>* __restrict__,                         \
                                real_type_t<PRECISION>* __restrict__,                         \
                                real_type_t<PRECISION>* __restrict__);                        \
                hipLaunchKernelGGL(kernel_func,                                               \
                                   dim3(data->gridParam.b_x),                                 \
                                   dim3(data->gridParam.tpb_x),                               \
                                   0,                                                         \
                                   rocfft_stream,                                             \
                                   (PRECISION*)data->node->twiddles.data(),                   \
                                   data->node->length.size(),                                 \
                                   data->node->devKernArg.data(),                             \
                                   data->node->devKernArg.data() + 1 * KERN_ARGS_ARRAY_WIDTH, \
                                   data->node->devKernArg.data() + 2 * KERN_ARGS_ARRAY_WIDTH, \
                                   data->node->batch,                                         \
                                   (real_type_t<PRECISION>*)data->bufIn[0],                   \
                                   (real_type_t<PRECISION>*)data->bufIn[1],                   \
                                   (real_type_t<PRECISION>*)data->bufOut[0],                  \
                                   (real_type_t<PRECISION>*)data->bufOut[1]);                 \
            }                                                                                 \
        }                                                                                     \
    }

#define POWX_LARGE_SBCC_GENERATOR(FUNCTION_NAME,                                              \
                                  IP_FWD_KERN_NAME,                                           \
                                  IP_BACK_KERN_NAME,                                          \
                                  OP_FWD_KERN_NAME,                                           \
                                  OP_BACK_KERN_NAME,                                          \
                                  PRECISION)                                                  \
    void FUNCTION_NAME(const void* data_p, void* back_p)                                      \
    {                                                                                         \
        DeviceCallIn* data          = (DeviceCallIn*)data_p;                                  \
        hipStream_t   rocfft_stream = data->rocfft_stream;                                    \
                                                                                              \
        /* The size of last dimension need to be counted into batch */                        \
        /* Check how to config thread block in PlanPowX() for SBCC  */                        \
        const size_t batch = (data->node->length.size() >= 3)                                 \
                                 ? data->node->batch * data->node->length.back()              \
                                 : data->node->batch;                                         \
                                                                                              \
        if(data->node->placement == rocfft_placement_inplace)                                 \
        {                                                                                     \
            if(array_type_is_interleaved(data->node->inArrayType)                             \
               && array_type_is_interleaved(data->node->outArrayType))                        \
            {                                                                                 \
                GET_KERNEL_FUNC_SBCC(IP_FWD_KERN_NAME,                                        \
                                     IP_BACK_KERN_NAME,                                       \
                                     PRECISION,                                               \
                                     KERNEL_BASE_ARGS_IP_SBCC,                                \
                                     PRECISION* __restrict__);                                \
                hipLaunchKernelGGL(kernel_func,                                               \
                                   dim3(data->gridParam.b_x),                                 \
                                   dim3(data->gridParam.tpb_x),                               \
                                   0,                                                         \
                                   rocfft_stream,                                             \
                                   (PRECISION*)data->node->twiddles.data(),                   \
                                   (PRECISION*)data->node->twiddles_large.data(),             \
                                   data->node->length.size(),                                 \
                                   data->node->devKernArg.data(),                             \
                                   data->node->devKernArg.data() + 1 * KERN_ARGS_ARRAY_WIDTH, \
                                   batch,                                                     \
                                   (PRECISION*)data->bufIn[0]);                               \
            }                                                                                 \
            else if(array_type_is_planar(data->node->inArrayType)                             \
                    && array_type_is_planar(data->node->outArrayType))                        \
            {                                                                                 \
                GET_KERNEL_FUNC_SBCC(IP_FWD_KERN_NAME,                                        \
                                     IP_BACK_KERN_NAME,                                       \
                                     PRECISION,                                               \
                                     KERNEL_BASE_ARGS_IP_SBCC,                                \
                                     real_type_t<PRECISION>* __restrict__,                    \
                                     real_type_t<PRECISION>* __restrict__);                   \
                hipLaunchKernelGGL(kernel_func,                                               \
                                   dim3(data->gridParam.b_x),                                 \
                                   dim3(data->gridParam.tpb_x),                               \
                                   0,                                                         \
                                   rocfft_stream,                                             \
                                   (PRECISION*)data->node->twiddles.data(),                   \
                                   (PRECISION*)data->node->twiddles_large.data(),             \
                                   data->node->length.size(),                                 \
                                   data->node->devKernArg.data(),                             \
                                   data->node->devKernArg.data() + 1 * KERN_ARGS_ARRAY_WIDTH, \
                                   batch,                                                     \
                                   (real_type_t<PRECISION>*)data->bufIn[0],                   \
                                   (real_type_t<PRECISION>*)data->bufIn[1]);                  \
            }                                                                                 \
        }                                                                                     \
        else                                                                                  \
        {                                                                                     \
            if(array_type_is_interleaved(data->node->inArrayType)                             \
               && array_type_is_interleaved(data->node->outArrayType))                        \
            {                                                                                 \
                GET_KERNEL_FUNC_SBCC(OP_FWD_KERN_NAME,                                        \
                                     OP_BACK_KERN_NAME,                                       \
                                     PRECISION,                                               \
                                     KERNEL_BASE_ARGS_OP_SBCC,                                \
                                     PRECISION* __restrict__,                                 \
                                     PRECISION* __restrict__);                                \
                hipLaunchKernelGGL(kernel_func,                                               \
                                   dim3(data->gridParam.b_x),                                 \
                                   dim3(data->gridParam.tpb_x),                               \
                                   0,                                                         \
                                   rocfft_stream,                                             \
                                   (PRECISION*)data->node->twiddles.data(),                   \
                                   (PRECISION*)data->node->twiddles_large.data(),             \
                                   data->node->length.size(),                                 \
                                   data->node->devKernArg.data(),                             \
                                   data->node->devKernArg.data() + 1 * KERN_ARGS_ARRAY_WIDTH, \
                                   data->node->devKernArg.data() + 2 * KERN_ARGS_ARRAY_WIDTH, \
                                   batch,                                                     \
                                   (PRECISION*)data->bufIn[0],                                \
                                   (PRECISION*)data->bufOut[0]);                              \
            }                                                                                 \
            else if(array_type_is_interleaved(data->node->inArrayType)                        \
                    && array_type_is_planar(data->node->outArrayType))                        \
            {                                                                                 \
                GET_KERNEL_FUNC_SBCC(OP_FWD_KERN_NAME,                                        \
                                     OP_BACK_KERN_NAME,                                       \
                                     PRECISION,                                               \
                                     KERNEL_BASE_ARGS_OP_SBCC,                                \
                                     PRECISION* __restrict__,                                 \
                                     real_type_t<PRECISION>* __restrict__,                    \
                                     real_type_t<PRECISION>* __restrict__);                   \
                hipLaunchKernelGGL(kernel_func,                                               \
                                   dim3(data->gridParam.b_x),                                 \
                                   dim3(data->gridParam.tpb_x),                               \
                                   0,                                                         \
                                   rocfft_stream,                                             \
                                   (PRECISION*)data->node->twiddles.data(),                   \
                                   (PRECISION*)data->node->twiddles_large.data(),             \
                                   data->node->length.size(),                                 \
                                   data->node->devKernArg.data(),                             \
                                   data->node->devKernArg.data() + 1 * KERN_ARGS_ARRAY_WIDTH, \
                                   data->node->devKernArg.data() + 2 * KERN_ARGS_ARRAY_WIDTH, \
                                   batch,                                                     \
                                   (PRECISION*)data->bufIn[0],                                \
                                   (real_type_t<PRECISION>*)data->bufOut[0],                  \
                                   (real_type_t<PRECISION>*)data->bufOut[1]);                 \
            }                                                                                 \
            else if(array_type_is_planar(data->node->inArrayType)                             \
                    && array_type_is_interleaved(data->node->outArrayType))                   \
            {                                                                                 \
                GET_KERNEL_FUNC_SBCC(OP_FWD_KERN_NAME,                                        \
                                     OP_BACK_KERN_NAME,                                       \
                                     PRECISION,                                               \
                                     KERNEL_BASE_ARGS_OP_SBCC,                                \
                                     real_type_t<PRECISION>* __restrict__,                    \
                                     real_type_t<PRECISION>* __restrict__,                    \
                                     PRECISION* __restrict__);                                \
                hipLaunchKernelGGL(kernel_func,                                               \
                                   dim3(data->gridParam.b_x),                                 \
                                   dim3(data->gridParam.tpb_x),                               \
                                   0,                                                         \
                                   rocfft_stream,                                             \
                                   (PRECISION*)data->node->twiddles.data(),                   \
                                   (PRECISION*)data->node->twiddles_large.data(),             \
                                   data->node->length.size(),                                 \
                                   data->node->devKernArg.data(),                             \
                                   data->node->devKernArg.data() + 1 * KERN_ARGS_ARRAY_WIDTH, \
                                   data->node->devKernArg.data() + 2 * KERN_ARGS_ARRAY_WIDTH, \
                                   batch,                                                     \
                                   (real_type_t<PRECISION>*)data->bufIn[0],                   \
                                   (real_type_t<PRECISION>*)data->bufIn[1],                   \
                                   (PRECISION*)data->bufOut[0]);                              \
            }                                                                                 \
            else if(array_type_is_planar(data->node->inArrayType)                             \
                    && array_type_is_planar(data->node->outArrayType))                        \
            {                                                                                 \
                GET_KERNEL_FUNC_SBCC(OP_FWD_KERN_NAME,                                        \
                                     OP_BACK_KERN_NAME,                                       \
                                     PRECISION,                                               \
                                     KERNEL_BASE_ARGS_OP_SBCC,                                \
                                     real_type_t<PRECISION>* __restrict__,                    \
                                     real_type_t<PRECISION>* __restrict__,                    \
                                     real_type_t<PRECISION>* __restrict__,                    \
                                     real_type_t<PRECISION>* __restrict__);                   \
                hipLaunchKernelGGL(kernel_func,                                               \
                                   dim3(data->gridParam.b_x),                                 \
                                   dim3(data->gridParam.tpb_x),                               \
                                   0,                                                         \
                                   rocfft_stream,                                             \
                                   (PRECISION*)data->node->twiddles.data(),                   \
                                   (PRECISION*)data->node->twiddles_large.data(),             \
                                   data->node->length.size(),                                 \
                                   data->node->devKernArg.data(),                             \
                                   data->node->devKernArg.data() + 1 * KERN_ARGS_ARRAY_WIDTH, \
                                   data->node->devKernArg.data() + 2 * KERN_ARGS_ARRAY_WIDTH, \
                                   batch,                                                     \
                                   (real_type_t<PRECISION>*)data->bufIn[0],                   \
                                   (real_type_t<PRECISION>*)data->bufIn[1],                   \
                                   (real_type_t<PRECISION>*)data->bufOut[0],                  \
                                   (real_type_t<PRECISION>*)data->bufOut[1]);                 \
            }                                                                                 \
        }                                                                                     \
    }

#define POWX_LARGE_SBRC_GENERATOR(                                                        \
    FUNCTION_NAME, FWD_KERN_NAME, BACK_KERN_NAME, PRECISION, COL_DIM, TRANSPOSE_TYPE)     \
    void FUNCTION_NAME(const void* data_p, void* back_p)                                  \
    {                                                                                     \
        DeviceCallIn* data          = (DeviceCallIn*)data_p;                              \
        hipStream_t   rocfft_stream = data->rocfft_stream;                                \
                                                                                          \
        /* The size of last dimension need to be counted into batch */                    \
        /* Check how to config thread block in PlanPowX() for SBRC  */                    \
        const size_t batch = (data->node->length.size() >= 3)                             \
                                 ? data->node->batch * data->node->length.back()          \
                                 : data->node->batch;                                     \
                                                                                          \
        if(array_type_is_interleaved(data->node->inArrayType)                             \
           && array_type_is_interleaved(data->node->outArrayType))                        \
        {                                                                                 \
            GET_KERNEL_FUNC_SBRC(FWD_KERN_NAME,                                           \
                                 BACK_KERN_NAME,                                          \
                                 PRECISION,                                               \
                                 COL_DIM,                                                 \
                                 TRANSPOSE_TYPE,                                          \
                                 KERNEL_BASE_ARGS_OP,                                     \
                                 PRECISION* __restrict__,                                 \
                                 PRECISION* __restrict__);                                \
            hipLaunchKernelGGL(kernel_func,                                               \
                               dim3(data->gridParam.b_x),                                 \
                               dim3(data->gridParam.tpb_x),                               \
                               0,                                                         \
                               rocfft_stream,                                             \
                               (PRECISION*)data->node->twiddles.data(),                   \
                               data->node->length.size(),                                 \
                               data->node->devKernArg.data(),                             \
                               data->node->devKernArg.data() + 1 * KERN_ARGS_ARRAY_WIDTH, \
                               data->node->devKernArg.data() + 2 * KERN_ARGS_ARRAY_WIDTH, \
                               batch,                                                     \
                               (PRECISION*)data->bufIn[0],                                \
                               (PRECISION*)data->bufOut[0]);                              \
        }                                                                                 \
        else if(array_type_is_interleaved(data->node->inArrayType)                        \
                && array_type_is_planar(data->node->outArrayType))                        \
        {                                                                                 \
            GET_KERNEL_FUNC_SBRC(FWD_KERN_NAME,                                           \
                                 BACK_KERN_NAME,                                          \
                                 PRECISION,                                               \
                                 COL_DIM,                                                 \
                                 TRANSPOSE_TYPE,                                          \
                                 KERNEL_BASE_ARGS_OP,                                     \
                                 PRECISION* __restrict__,                                 \
                                 real_type_t<PRECISION>* __restrict__,                    \
                                 real_type_t<PRECISION>* __restrict__);                   \
            hipLaunchKernelGGL(kernel_func,                                               \
                               dim3(data->gridParam.b_x),                                 \
                               dim3(data->gridParam.tpb_x),                               \
                               0,                                                         \
                               rocfft_stream,                                             \
                               (PRECISION*)data->node->twiddles.data(),                   \
                               data->node->length.size(),                                 \
                               data->node->devKernArg.data(),                             \
                               data->node->devKernArg.data() + 1 * KERN_ARGS_ARRAY_WIDTH, \
                               data->node->devKernArg.data() + 2 * KERN_ARGS_ARRAY_WIDTH, \
                               batch,                                                     \
                               (PRECISION*)data->bufIn[0],                                \
                               (real_type_t<PRECISION>*)data->bufOut[0],                  \
                               (real_type_t<PRECISION>*)data->bufOut[1]);                 \
        }                                                                                 \
        else if(array_type_is_planar(data->node->inArrayType)                             \
                && array_type_is_interleaved(data->node->outArrayType))                   \
        {                                                                                 \
            GET_KERNEL_FUNC_SBRC(FWD_KERN_NAME,                                           \
                                 BACK_KERN_NAME,                                          \
                                 PRECISION,                                               \
                                 COL_DIM,                                                 \
                                 TRANSPOSE_TYPE,                                          \
                                 KERNEL_BASE_ARGS_OP,                                     \
                                 real_type_t<PRECISION>* __restrict__,                    \
                                 real_type_t<PRECISION>* __restrict__,                    \
                                 PRECISION* __restrict__);                                \
            hipLaunchKernelGGL(kernel_func,                                               \
                               dim3(data->gridParam.b_x),                                 \
                               dim3(data->gridParam.tpb_x),                               \
                               0,                                                         \
                               rocfft_stream,                                             \
                               (PRECISION*)data->node->twiddles.data(),                   \
                               data->node->length.size(),                                 \
                               data->node->devKernArg.data(),                             \
                               data->node->devKernArg.data() + 1 * KERN_ARGS_ARRAY_WIDTH, \
                               data->node->devKernArg.data() + 2 * KERN_ARGS_ARRAY_WIDTH, \
                               batch,                                                     \
                               (real_type_t<PRECISION>*)data->bufIn[0],                   \
                               (real_type_t<PRECISION>*)data->bufIn[1],                   \
                               (PRECISION*)data->bufOut[0]);                              \
        }                                                                                 \
        else if(array_type_is_planar(data->node->inArrayType)                             \
                && array_type_is_planar(data->node->outArrayType))                        \
        {                                                                                 \
            GET_KERNEL_FUNC_SBRC(FWD_KERN_NAME,                                           \
                                 BACK_KERN_NAME,                                          \
                                 PRECISION,                                               \
                                 COL_DIM,                                                 \
                                 TRANSPOSE_TYPE,                                          \
                                 KERNEL_BASE_ARGS_OP,                                     \
                                 real_type_t<PRECISION>* __restrict__,                    \
                                 real_type_t<PRECISION>* __restrict__,                    \
                                 real_type_t<PRECISION>* __restrict__,                    \
                                 real_type_t<PRECISION>* __restrict__);                   \
            hipLaunchKernelGGL(kernel_func,                                               \
                               dim3(data->gridParam.b_x),                                 \
                               dim3(data->gridParam.tpb_x),                               \
                               0,                                                         \
                               rocfft_stream,                                             \
                               (PRECISION*)data->node->twiddles.data(),                   \
                               data->node->length.size(),                                 \
                               data->node->devKernArg.data(),                             \
                               data->node->devKernArg.data() + 1 * KERN_ARGS_ARRAY_WIDTH, \
                               data->node->devKernArg.data() + 2 * KERN_ARGS_ARRAY_WIDTH, \
                               batch,                                                     \
                               (real_type_t<PRECISION>*)data->bufIn[0],                   \
                               (real_type_t<PRECISION>*)data->bufIn[1],                   \
                               (real_type_t<PRECISION>*)data->bufOut[0],                  \
                               (real_type_t<PRECISION>*)data->bufOut[1]);                 \
        }                                                                                 \
    }

#endif // KERNEL_LAUNCH_SINGLE
