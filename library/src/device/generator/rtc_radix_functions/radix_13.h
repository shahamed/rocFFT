/*******************************************************************************
 * Copyright (C) 2016-2022 Advanced Micro Devices, Inc. All rights reserved.
 ******************************************************************************/

template <typename T>
__device__ void FwdRad13B1(
    T* R0, T* R1, T* R2, T* R3, T* R4, T* R5, T* R6, T* R7, T* R8, T* R9, T* R10, T* R11, T* R12)
{
    T x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, dp, dm;

    x0 = (*R0) + (*R1) + (*R2) + (*R3) + (*R4) + (*R5) + (*R6) + (*R7) + (*R8) + (*R9) + (*R10)
         + (*R11) + (*R12);
    x1  = (*R0);
    x2  = (*R0);
    x3  = (*R0);
    x4  = (*R0);
    x5  = (*R0);
    x6  = (*R0);
    x7  = (*R0);
    x8  = (*R0);
    x9  = (*R0);
    x10 = (*R0);
    x11 = (*R0);
    x12 = (*R0);
    dp  = (*R1) + (*R12);
    dm  = (*R1) - (*R12);
    x1.x += Q13i1j1R * dp.x - Q13i1j1I * dm.y;
    x1.y += Q13i1j1R * dp.y + Q13i1j1I * dm.x;
    x12.x += Q13i1j1R * dp.x + Q13i1j1I * dm.y;
    x12.y += Q13i1j1R * dp.y - Q13i1j1I * dm.x;
    x2.x += Q13i2j1R * dp.x - Q13i2j1I * dm.y;
    x2.y += Q13i2j1R * dp.y + Q13i2j1I * dm.x;
    x11.x += Q13i2j1R * dp.x + Q13i2j1I * dm.y;
    x11.y += Q13i2j1R * dp.y - Q13i2j1I * dm.x;
    x3.x += Q13i3j1R * dp.x - Q13i3j1I * dm.y;
    x3.y += Q13i3j1R * dp.y + Q13i3j1I * dm.x;
    x10.x += Q13i3j1R * dp.x + Q13i3j1I * dm.y;
    x10.y += Q13i3j1R * dp.y - Q13i3j1I * dm.x;
    x4.x += Q13i4j1R * dp.x - Q13i4j1I * dm.y;
    x4.y += Q13i4j1R * dp.y + Q13i4j1I * dm.x;
    x9.x += Q13i4j1R * dp.x + Q13i4j1I * dm.y;
    x9.y += Q13i4j1R * dp.y - Q13i4j1I * dm.x;
    x5.x += Q13i5j1R * dp.x - Q13i5j1I * dm.y;
    x5.y += Q13i5j1R * dp.y + Q13i5j1I * dm.x;
    x8.x += Q13i5j1R * dp.x + Q13i5j1I * dm.y;
    x8.y += Q13i5j1R * dp.y - Q13i5j1I * dm.x;
    x6.x += Q13i6j1R * dp.x - Q13i6j1I * dm.y;
    x6.y += Q13i6j1R * dp.y + Q13i6j1I * dm.x;
    x7.x += Q13i6j1R * dp.x + Q13i6j1I * dm.y;
    x7.y += Q13i6j1R * dp.y - Q13i6j1I * dm.x;
    dp = (*R2) + (*R11);
    dm = (*R2) - (*R11);
    x1.x += Q13i1j2R * dp.x - Q13i1j2I * dm.y;
    x1.y += Q13i1j2R * dp.y + Q13i1j2I * dm.x;
    x12.x += Q13i1j2R * dp.x + Q13i1j2I * dm.y;
    x12.y += Q13i1j2R * dp.y - Q13i1j2I * dm.x;
    x2.x += Q13i2j2R * dp.x - Q13i2j2I * dm.y;
    x2.y += Q13i2j2R * dp.y + Q13i2j2I * dm.x;
    x11.x += Q13i2j2R * dp.x + Q13i2j2I * dm.y;
    x11.y += Q13i2j2R * dp.y - Q13i2j2I * dm.x;
    x3.x += Q13i3j2R * dp.x - Q13i3j2I * dm.y;
    x3.y += Q13i3j2R * dp.y + Q13i3j2I * dm.x;
    x10.x += Q13i3j2R * dp.x + Q13i3j2I * dm.y;
    x10.y += Q13i3j2R * dp.y - Q13i3j2I * dm.x;
    x4.x += Q13i4j2R * dp.x - Q13i4j2I * dm.y;
    x4.y += Q13i4j2R * dp.y + Q13i4j2I * dm.x;
    x9.x += Q13i4j2R * dp.x + Q13i4j2I * dm.y;
    x9.y += Q13i4j2R * dp.y - Q13i4j2I * dm.x;
    x5.x += Q13i5j2R * dp.x - Q13i5j2I * dm.y;
    x5.y += Q13i5j2R * dp.y + Q13i5j2I * dm.x;
    x8.x += Q13i5j2R * dp.x + Q13i5j2I * dm.y;
    x8.y += Q13i5j2R * dp.y - Q13i5j2I * dm.x;
    x6.x += Q13i6j2R * dp.x - Q13i6j2I * dm.y;
    x6.y += Q13i6j2R * dp.y + Q13i6j2I * dm.x;
    x7.x += Q13i6j2R * dp.x + Q13i6j2I * dm.y;
    x7.y += Q13i6j2R * dp.y - Q13i6j2I * dm.x;
    dp = (*R3) + (*R10);
    dm = (*R3) - (*R10);
    x1.x += Q13i1j3R * dp.x - Q13i1j3I * dm.y;
    x1.y += Q13i1j3R * dp.y + Q13i1j3I * dm.x;
    x12.x += Q13i1j3R * dp.x + Q13i1j3I * dm.y;
    x12.y += Q13i1j3R * dp.y - Q13i1j3I * dm.x;
    x2.x += Q13i2j3R * dp.x - Q13i2j3I * dm.y;
    x2.y += Q13i2j3R * dp.y + Q13i2j3I * dm.x;
    x11.x += Q13i2j3R * dp.x + Q13i2j3I * dm.y;
    x11.y += Q13i2j3R * dp.y - Q13i2j3I * dm.x;
    x3.x += Q13i3j3R * dp.x - Q13i3j3I * dm.y;
    x3.y += Q13i3j3R * dp.y + Q13i3j3I * dm.x;
    x10.x += Q13i3j3R * dp.x + Q13i3j3I * dm.y;
    x10.y += Q13i3j3R * dp.y - Q13i3j3I * dm.x;
    x4.x += Q13i4j3R * dp.x - Q13i4j3I * dm.y;
    x4.y += Q13i4j3R * dp.y + Q13i4j3I * dm.x;
    x9.x += Q13i4j3R * dp.x + Q13i4j3I * dm.y;
    x9.y += Q13i4j3R * dp.y - Q13i4j3I * dm.x;
    x5.x += Q13i5j3R * dp.x - Q13i5j3I * dm.y;
    x5.y += Q13i5j3R * dp.y + Q13i5j3I * dm.x;
    x8.x += Q13i5j3R * dp.x + Q13i5j3I * dm.y;
    x8.y += Q13i5j3R * dp.y - Q13i5j3I * dm.x;
    x6.x += Q13i6j3R * dp.x - Q13i6j3I * dm.y;
    x6.y += Q13i6j3R * dp.y + Q13i6j3I * dm.x;
    x7.x += Q13i6j3R * dp.x + Q13i6j3I * dm.y;
    x7.y += Q13i6j3R * dp.y - Q13i6j3I * dm.x;
    dp = (*R4) + (*R9);
    dm = (*R4) - (*R9);
    x1.x += Q13i1j4R * dp.x - Q13i1j4I * dm.y;
    x1.y += Q13i1j4R * dp.y + Q13i1j4I * dm.x;
    x12.x += Q13i1j4R * dp.x + Q13i1j4I * dm.y;
    x12.y += Q13i1j4R * dp.y - Q13i1j4I * dm.x;
    x2.x += Q13i2j4R * dp.x - Q13i2j4I * dm.y;
    x2.y += Q13i2j4R * dp.y + Q13i2j4I * dm.x;
    x11.x += Q13i2j4R * dp.x + Q13i2j4I * dm.y;
    x11.y += Q13i2j4R * dp.y - Q13i2j4I * dm.x;
    x3.x += Q13i3j4R * dp.x - Q13i3j4I * dm.y;
    x3.y += Q13i3j4R * dp.y + Q13i3j4I * dm.x;
    x10.x += Q13i3j4R * dp.x + Q13i3j4I * dm.y;
    x10.y += Q13i3j4R * dp.y - Q13i3j4I * dm.x;
    x4.x += Q13i4j4R * dp.x - Q13i4j4I * dm.y;
    x4.y += Q13i4j4R * dp.y + Q13i4j4I * dm.x;
    x9.x += Q13i4j4R * dp.x + Q13i4j4I * dm.y;
    x9.y += Q13i4j4R * dp.y - Q13i4j4I * dm.x;
    x5.x += Q13i5j4R * dp.x - Q13i5j4I * dm.y;
    x5.y += Q13i5j4R * dp.y + Q13i5j4I * dm.x;
    x8.x += Q13i5j4R * dp.x + Q13i5j4I * dm.y;
    x8.y += Q13i5j4R * dp.y - Q13i5j4I * dm.x;
    x6.x += Q13i6j4R * dp.x - Q13i6j4I * dm.y;
    x6.y += Q13i6j4R * dp.y + Q13i6j4I * dm.x;
    x7.x += Q13i6j4R * dp.x + Q13i6j4I * dm.y;
    x7.y += Q13i6j4R * dp.y - Q13i6j4I * dm.x;
    dp = (*R5) + (*R8);
    dm = (*R5) - (*R8);
    x1.x += Q13i1j5R * dp.x - Q13i1j5I * dm.y;
    x1.y += Q13i1j5R * dp.y + Q13i1j5I * dm.x;
    x12.x += Q13i1j5R * dp.x + Q13i1j5I * dm.y;
    x12.y += Q13i1j5R * dp.y - Q13i1j5I * dm.x;
    x2.x += Q13i2j5R * dp.x - Q13i2j5I * dm.y;
    x2.y += Q13i2j5R * dp.y + Q13i2j5I * dm.x;
    x11.x += Q13i2j5R * dp.x + Q13i2j5I * dm.y;
    x11.y += Q13i2j5R * dp.y - Q13i2j5I * dm.x;
    x3.x += Q13i3j5R * dp.x - Q13i3j5I * dm.y;
    x3.y += Q13i3j5R * dp.y + Q13i3j5I * dm.x;
    x10.x += Q13i3j5R * dp.x + Q13i3j5I * dm.y;
    x10.y += Q13i3j5R * dp.y - Q13i3j5I * dm.x;
    x4.x += Q13i4j5R * dp.x - Q13i4j5I * dm.y;
    x4.y += Q13i4j5R * dp.y + Q13i4j5I * dm.x;
    x9.x += Q13i4j5R * dp.x + Q13i4j5I * dm.y;
    x9.y += Q13i4j5R * dp.y - Q13i4j5I * dm.x;
    x5.x += Q13i5j5R * dp.x - Q13i5j5I * dm.y;
    x5.y += Q13i5j5R * dp.y + Q13i5j5I * dm.x;
    x8.x += Q13i5j5R * dp.x + Q13i5j5I * dm.y;
    x8.y += Q13i5j5R * dp.y - Q13i5j5I * dm.x;
    x6.x += Q13i6j5R * dp.x - Q13i6j5I * dm.y;
    x6.y += Q13i6j5R * dp.y + Q13i6j5I * dm.x;
    x7.x += Q13i6j5R * dp.x + Q13i6j5I * dm.y;
    x7.y += Q13i6j5R * dp.y - Q13i6j5I * dm.x;
    dp = (*R6) + (*R7);
    dm = (*R6) - (*R7);
    x1.x += Q13i1j6R * dp.x - Q13i1j6I * dm.y;
    x1.y += Q13i1j6R * dp.y + Q13i1j6I * dm.x;
    x12.x += Q13i1j6R * dp.x + Q13i1j6I * dm.y;
    x12.y += Q13i1j6R * dp.y - Q13i1j6I * dm.x;
    x2.x += Q13i2j6R * dp.x - Q13i2j6I * dm.y;
    x2.y += Q13i2j6R * dp.y + Q13i2j6I * dm.x;
    x11.x += Q13i2j6R * dp.x + Q13i2j6I * dm.y;
    x11.y += Q13i2j6R * dp.y - Q13i2j6I * dm.x;
    x3.x += Q13i3j6R * dp.x - Q13i3j6I * dm.y;
    x3.y += Q13i3j6R * dp.y + Q13i3j6I * dm.x;
    x10.x += Q13i3j6R * dp.x + Q13i3j6I * dm.y;
    x10.y += Q13i3j6R * dp.y - Q13i3j6I * dm.x;
    x4.x += Q13i4j6R * dp.x - Q13i4j6I * dm.y;
    x4.y += Q13i4j6R * dp.y + Q13i4j6I * dm.x;
    x9.x += Q13i4j6R * dp.x + Q13i4j6I * dm.y;
    x9.y += Q13i4j6R * dp.y - Q13i4j6I * dm.x;
    x5.x += Q13i5j6R * dp.x - Q13i5j6I * dm.y;
    x5.y += Q13i5j6R * dp.y + Q13i5j6I * dm.x;
    x8.x += Q13i5j6R * dp.x + Q13i5j6I * dm.y;
    x8.y += Q13i5j6R * dp.y - Q13i5j6I * dm.x;
    x6.x += Q13i6j6R * dp.x - Q13i6j6I * dm.y;
    x6.y += Q13i6j6R * dp.y + Q13i6j6I * dm.x;
    x7.x += Q13i6j6R * dp.x + Q13i6j6I * dm.y;
    x7.y += Q13i6j6R * dp.y - Q13i6j6I * dm.x;
    (*R0)  = x0;
    (*R1)  = x1;
    (*R2)  = x2;
    (*R3)  = x3;
    (*R4)  = x4;
    (*R5)  = x5;
    (*R6)  = x6;
    (*R7)  = x7;
    (*R8)  = x8;
    (*R9)  = x9;
    (*R10) = x10;
    (*R11) = x11;
    (*R12) = x12;
}

template <typename T>
__device__ void InvRad13B1(
    T* R0, T* R1, T* R2, T* R3, T* R4, T* R5, T* R6, T* R7, T* R8, T* R9, T* R10, T* R11, T* R12)
{
    T x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, dp, dm;

    x0 = (*R0) + (*R1) + (*R2) + (*R3) + (*R4) + (*R5) + (*R6) + (*R7) + (*R8) + (*R9) + (*R10)
         + (*R11) + (*R12);
    x1  = (*R0);
    x2  = (*R0);
    x3  = (*R0);
    x4  = (*R0);
    x5  = (*R0);
    x6  = (*R0);
    x7  = (*R0);
    x8  = (*R0);
    x9  = (*R0);
    x10 = (*R0);
    x11 = (*R0);
    x12 = (*R0);
    dp  = (*R1) + (*R12);
    dm  = (*R1) - (*R12);
    x1.x += Q13i1j1R * dp.x + Q13i1j1I * dm.y;
    x1.y += Q13i1j1R * dp.y - Q13i1j1I * dm.x;
    x12.x += Q13i1j1R * dp.x - Q13i1j1I * dm.y;
    x12.y += Q13i1j1R * dp.y + Q13i1j1I * dm.x;
    x2.x += Q13i2j1R * dp.x + Q13i2j1I * dm.y;
    x2.y += Q13i2j1R * dp.y - Q13i2j1I * dm.x;
    x11.x += Q13i2j1R * dp.x - Q13i2j1I * dm.y;
    x11.y += Q13i2j1R * dp.y + Q13i2j1I * dm.x;
    x3.x += Q13i3j1R * dp.x + Q13i3j1I * dm.y;
    x3.y += Q13i3j1R * dp.y - Q13i3j1I * dm.x;
    x10.x += Q13i3j1R * dp.x - Q13i3j1I * dm.y;
    x10.y += Q13i3j1R * dp.y + Q13i3j1I * dm.x;
    x4.x += Q13i4j1R * dp.x + Q13i4j1I * dm.y;
    x4.y += Q13i4j1R * dp.y - Q13i4j1I * dm.x;
    x9.x += Q13i4j1R * dp.x - Q13i4j1I * dm.y;
    x9.y += Q13i4j1R * dp.y + Q13i4j1I * dm.x;
    x5.x += Q13i5j1R * dp.x + Q13i5j1I * dm.y;
    x5.y += Q13i5j1R * dp.y - Q13i5j1I * dm.x;
    x8.x += Q13i5j1R * dp.x - Q13i5j1I * dm.y;
    x8.y += Q13i5j1R * dp.y + Q13i5j1I * dm.x;
    x6.x += Q13i6j1R * dp.x + Q13i6j1I * dm.y;
    x6.y += Q13i6j1R * dp.y - Q13i6j1I * dm.x;
    x7.x += Q13i6j1R * dp.x - Q13i6j1I * dm.y;
    x7.y += Q13i6j1R * dp.y + Q13i6j1I * dm.x;
    dp = (*R2) + (*R11);
    dm = (*R2) - (*R11);
    x1.x += Q13i1j2R * dp.x + Q13i1j2I * dm.y;
    x1.y += Q13i1j2R * dp.y - Q13i1j2I * dm.x;
    x12.x += Q13i1j2R * dp.x - Q13i1j2I * dm.y;
    x12.y += Q13i1j2R * dp.y + Q13i1j2I * dm.x;
    x2.x += Q13i2j2R * dp.x + Q13i2j2I * dm.y;
    x2.y += Q13i2j2R * dp.y - Q13i2j2I * dm.x;
    x11.x += Q13i2j2R * dp.x - Q13i2j2I * dm.y;
    x11.y += Q13i2j2R * dp.y + Q13i2j2I * dm.x;
    x3.x += Q13i3j2R * dp.x + Q13i3j2I * dm.y;
    x3.y += Q13i3j2R * dp.y - Q13i3j2I * dm.x;
    x10.x += Q13i3j2R * dp.x - Q13i3j2I * dm.y;
    x10.y += Q13i3j2R * dp.y + Q13i3j2I * dm.x;
    x4.x += Q13i4j2R * dp.x + Q13i4j2I * dm.y;
    x4.y += Q13i4j2R * dp.y - Q13i4j2I * dm.x;
    x9.x += Q13i4j2R * dp.x - Q13i4j2I * dm.y;
    x9.y += Q13i4j2R * dp.y + Q13i4j2I * dm.x;
    x5.x += Q13i5j2R * dp.x + Q13i5j2I * dm.y;
    x5.y += Q13i5j2R * dp.y - Q13i5j2I * dm.x;
    x8.x += Q13i5j2R * dp.x - Q13i5j2I * dm.y;
    x8.y += Q13i5j2R * dp.y + Q13i5j2I * dm.x;
    x6.x += Q13i6j2R * dp.x + Q13i6j2I * dm.y;
    x6.y += Q13i6j2R * dp.y - Q13i6j2I * dm.x;
    x7.x += Q13i6j2R * dp.x - Q13i6j2I * dm.y;
    x7.y += Q13i6j2R * dp.y + Q13i6j2I * dm.x;
    dp = (*R3) + (*R10);
    dm = (*R3) - (*R10);
    x1.x += Q13i1j3R * dp.x + Q13i1j3I * dm.y;
    x1.y += Q13i1j3R * dp.y - Q13i1j3I * dm.x;
    x12.x += Q13i1j3R * dp.x - Q13i1j3I * dm.y;
    x12.y += Q13i1j3R * dp.y + Q13i1j3I * dm.x;
    x2.x += Q13i2j3R * dp.x + Q13i2j3I * dm.y;
    x2.y += Q13i2j3R * dp.y - Q13i2j3I * dm.x;
    x11.x += Q13i2j3R * dp.x - Q13i2j3I * dm.y;
    x11.y += Q13i2j3R * dp.y + Q13i2j3I * dm.x;
    x3.x += Q13i3j3R * dp.x + Q13i3j3I * dm.y;
    x3.y += Q13i3j3R * dp.y - Q13i3j3I * dm.x;
    x10.x += Q13i3j3R * dp.x - Q13i3j3I * dm.y;
    x10.y += Q13i3j3R * dp.y + Q13i3j3I * dm.x;
    x4.x += Q13i4j3R * dp.x + Q13i4j3I * dm.y;
    x4.y += Q13i4j3R * dp.y - Q13i4j3I * dm.x;
    x9.x += Q13i4j3R * dp.x - Q13i4j3I * dm.y;
    x9.y += Q13i4j3R * dp.y + Q13i4j3I * dm.x;
    x5.x += Q13i5j3R * dp.x + Q13i5j3I * dm.y;
    x5.y += Q13i5j3R * dp.y - Q13i5j3I * dm.x;
    x8.x += Q13i5j3R * dp.x - Q13i5j3I * dm.y;
    x8.y += Q13i5j3R * dp.y + Q13i5j3I * dm.x;
    x6.x += Q13i6j3R * dp.x + Q13i6j3I * dm.y;
    x6.y += Q13i6j3R * dp.y - Q13i6j3I * dm.x;
    x7.x += Q13i6j3R * dp.x - Q13i6j3I * dm.y;
    x7.y += Q13i6j3R * dp.y + Q13i6j3I * dm.x;
    dp = (*R4) + (*R9);
    dm = (*R4) - (*R9);
    x1.x += Q13i1j4R * dp.x + Q13i1j4I * dm.y;
    x1.y += Q13i1j4R * dp.y - Q13i1j4I * dm.x;
    x12.x += Q13i1j4R * dp.x - Q13i1j4I * dm.y;
    x12.y += Q13i1j4R * dp.y + Q13i1j4I * dm.x;
    x2.x += Q13i2j4R * dp.x + Q13i2j4I * dm.y;
    x2.y += Q13i2j4R * dp.y - Q13i2j4I * dm.x;
    x11.x += Q13i2j4R * dp.x - Q13i2j4I * dm.y;
    x11.y += Q13i2j4R * dp.y + Q13i2j4I * dm.x;
    x3.x += Q13i3j4R * dp.x + Q13i3j4I * dm.y;
    x3.y += Q13i3j4R * dp.y - Q13i3j4I * dm.x;
    x10.x += Q13i3j4R * dp.x - Q13i3j4I * dm.y;
    x10.y += Q13i3j4R * dp.y + Q13i3j4I * dm.x;
    x4.x += Q13i4j4R * dp.x + Q13i4j4I * dm.y;
    x4.y += Q13i4j4R * dp.y - Q13i4j4I * dm.x;
    x9.x += Q13i4j4R * dp.x - Q13i4j4I * dm.y;
    x9.y += Q13i4j4R * dp.y + Q13i4j4I * dm.x;
    x5.x += Q13i5j4R * dp.x + Q13i5j4I * dm.y;
    x5.y += Q13i5j4R * dp.y - Q13i5j4I * dm.x;
    x8.x += Q13i5j4R * dp.x - Q13i5j4I * dm.y;
    x8.y += Q13i5j4R * dp.y + Q13i5j4I * dm.x;
    x6.x += Q13i6j4R * dp.x + Q13i6j4I * dm.y;
    x6.y += Q13i6j4R * dp.y - Q13i6j4I * dm.x;
    x7.x += Q13i6j4R * dp.x - Q13i6j4I * dm.y;
    x7.y += Q13i6j4R * dp.y + Q13i6j4I * dm.x;
    dp = (*R5) + (*R8);
    dm = (*R5) - (*R8);
    x1.x += Q13i1j5R * dp.x + Q13i1j5I * dm.y;
    x1.y += Q13i1j5R * dp.y - Q13i1j5I * dm.x;
    x12.x += Q13i1j5R * dp.x - Q13i1j5I * dm.y;
    x12.y += Q13i1j5R * dp.y + Q13i1j5I * dm.x;
    x2.x += Q13i2j5R * dp.x + Q13i2j5I * dm.y;
    x2.y += Q13i2j5R * dp.y - Q13i2j5I * dm.x;
    x11.x += Q13i2j5R * dp.x - Q13i2j5I * dm.y;
    x11.y += Q13i2j5R * dp.y + Q13i2j5I * dm.x;
    x3.x += Q13i3j5R * dp.x + Q13i3j5I * dm.y;
    x3.y += Q13i3j5R * dp.y - Q13i3j5I * dm.x;
    x10.x += Q13i3j5R * dp.x - Q13i3j5I * dm.y;
    x10.y += Q13i3j5R * dp.y + Q13i3j5I * dm.x;
    x4.x += Q13i4j5R * dp.x + Q13i4j5I * dm.y;
    x4.y += Q13i4j5R * dp.y - Q13i4j5I * dm.x;
    x9.x += Q13i4j5R * dp.x - Q13i4j5I * dm.y;
    x9.y += Q13i4j5R * dp.y + Q13i4j5I * dm.x;
    x5.x += Q13i5j5R * dp.x + Q13i5j5I * dm.y;
    x5.y += Q13i5j5R * dp.y - Q13i5j5I * dm.x;
    x8.x += Q13i5j5R * dp.x - Q13i5j5I * dm.y;
    x8.y += Q13i5j5R * dp.y + Q13i5j5I * dm.x;
    x6.x += Q13i6j5R * dp.x + Q13i6j5I * dm.y;
    x6.y += Q13i6j5R * dp.y - Q13i6j5I * dm.x;
    x7.x += Q13i6j5R * dp.x - Q13i6j5I * dm.y;
    x7.y += Q13i6j5R * dp.y + Q13i6j5I * dm.x;
    dp = (*R6) + (*R7);
    dm = (*R6) - (*R7);
    x1.x += Q13i1j6R * dp.x + Q13i1j6I * dm.y;
    x1.y += Q13i1j6R * dp.y - Q13i1j6I * dm.x;
    x12.x += Q13i1j6R * dp.x - Q13i1j6I * dm.y;
    x12.y += Q13i1j6R * dp.y + Q13i1j6I * dm.x;
    x2.x += Q13i2j6R * dp.x + Q13i2j6I * dm.y;
    x2.y += Q13i2j6R * dp.y - Q13i2j6I * dm.x;
    x11.x += Q13i2j6R * dp.x - Q13i2j6I * dm.y;
    x11.y += Q13i2j6R * dp.y + Q13i2j6I * dm.x;
    x3.x += Q13i3j6R * dp.x + Q13i3j6I * dm.y;
    x3.y += Q13i3j6R * dp.y - Q13i3j6I * dm.x;
    x10.x += Q13i3j6R * dp.x - Q13i3j6I * dm.y;
    x10.y += Q13i3j6R * dp.y + Q13i3j6I * dm.x;
    x4.x += Q13i4j6R * dp.x + Q13i4j6I * dm.y;
    x4.y += Q13i4j6R * dp.y - Q13i4j6I * dm.x;
    x9.x += Q13i4j6R * dp.x - Q13i4j6I * dm.y;
    x9.y += Q13i4j6R * dp.y + Q13i4j6I * dm.x;
    x5.x += Q13i5j6R * dp.x + Q13i5j6I * dm.y;
    x5.y += Q13i5j6R * dp.y - Q13i5j6I * dm.x;
    x8.x += Q13i5j6R * dp.x - Q13i5j6I * dm.y;
    x8.y += Q13i5j6R * dp.y + Q13i5j6I * dm.x;
    x6.x += Q13i6j6R * dp.x + Q13i6j6I * dm.y;
    x6.y += Q13i6j6R * dp.y - Q13i6j6I * dm.x;
    x7.x += Q13i6j6R * dp.x - Q13i6j6I * dm.y;
    x7.y += Q13i6j6R * dp.y + Q13i6j6I * dm.x;
    (*R0)  = x0;
    (*R1)  = x1;
    (*R2)  = x2;
    (*R3)  = x3;
    (*R4)  = x4;
    (*R5)  = x5;
    (*R6)  = x6;
    (*R7)  = x7;
    (*R8)  = x8;
    (*R9)  = x9;
    (*R10) = x10;
    (*R11) = x11;
    (*R12) = x12;
}