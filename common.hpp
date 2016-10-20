/*
 * Copyright (c) 2014, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in 
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#pragma once

#define ASSERT(x)                                                       \
    do                                                                  \
        {                                                               \
            if (!(x))                                                   \
                {                                                       \
                    fprintf(stdout, "Assertion \"%s\" failed at %s:%d\n", #x, __FILE__, __LINE__); \
                    /*exit(EXIT_FAILURE);*/                                 \
                }                                                       \
        } while (0)

#define ASSERTDRV(stmt)				\
    do                                          \
        {                                       \
            CUresult result = (stmt);           \
            ASSERT(CUDA_SUCCESS == result);     \
        } while (0)

#define ASSERTRT(stmt)				\
    do                                          \
        {                                       \
            cudaError_t result = (stmt);           \
            ASSERT(cudaSuccess == result);     \
        } while (0)

#define ASSERT_EQ(P, V) ASSERT((P) == (V))
#define CHECK_EQ(P, V) ASSERT((P) == (V))
#define ASSERT_NEQ(P, V) ASSERT((P) != (V))
#define BREAK_IF_NEQ(P, V) if((P) != (V)) break
#define BEGIN_CHECK do
#define END_CHECK while(0)

static void compare_buf(uint32_t *ref_buf, uint32_t *buf, size_t size)
{
    int diff = 0;
    ASSERT_EQ(size % 4, 0U);
    for(unsigned  w = 0; w<size/sizeof(uint32_t); ++w) {
		if (ref_buf[w] != buf[w]) { 
			if (diff < 10)
				printf("[word %d] %08x != %08x\n", w, buf[w], ref_buf[w]);
			++diff;
		}
    }
    //OUT << "diff(s): " << diff << endl;
    //CHECK_EQ(diff, 0);
    if (diff) {
        cout << "check error: diff(s)=" << diff << endl;
    }
}

static void init_hbuf_walking_bit(uint32_t *h_buf, size_t size)
{
    uint32_t base_value = 0x3F4C5E6A; // 0xa55ad33d;
    unsigned w;
    ASSERT_NEQ(h_buf, (void*)0);
    ASSERT_EQ(size % 4, 0U);
    //OUT << "filling mem with walking bit " << endl;
    for(w = 0; w<size/sizeof(uint32_t); ++w)
        h_buf[w] = base_value ^ (1<< (w%32));
}

static void init_hbuf_linear_ramp(uint32_t *h_buf, size_t size)
{
    uint32_t base_value = 0x3F4C5E6A; // 0xa55ad33d;
    unsigned w;
    ASSERT_NEQ(h_buf, (void*)0);
    ASSERT_EQ(size % 4, 0U);
    //OUT << "filling mem with walking bit " << endl;
    for(w = 0; w<size/sizeof(uint32_t); ++w)
        h_buf[w] = w;
}

