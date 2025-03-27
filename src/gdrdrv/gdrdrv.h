/*
 * Copyright (c) 2014-2021, NVIDIA CORPORATION. All rights reserved.
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

#ifndef __GDR_DRV_H__
#define __GDR_DRV_H__

#define GDRDRV_STRINGIFY(s)           #s
#define GDRDRV_TOSTRING(s)            GDRDRV_STRINGIFY(s)

#define GDRDRV_MAJOR_VERSION_SHIFT    16

#define GDRDRV_MAJOR_VERSION    2
#define GDRDRV_MINOR_VERSION    5
#define GDRDRV_VERSION          ((GDRDRV_MAJOR_VERSION << GDRDRV_MAJOR_VERSION_SHIFT) | GDRDRV_MINOR_VERSION)
#define GDRDRV_VERSION_STRING   GDRDRV_TOSTRING(GDRDRV_MAJOR_VERSION) "." GDRDRV_TOSTRING(GDRDRV_MINOR_VERSION)

#define MINIMUM_GDR_API_MAJOR_VERSION   2
#define MINIMUM_GDR_API_MINOR_VERSION   0
#define MINIMUM_GDR_API_VERSION         ((MINIMUM_GDR_API_MAJOR_VERSION << 16) | MINIMUM_GDR_API_MINOR_VERSION)

#define GDRDRV_MINIMUM_VERSION_WITH_GET_INFO_V2 ((2 << GDRDRV_MAJOR_VERSION_SHIFT) | 4)
#define GDRDRV_MINIMUM_VERSION_WITH_GET_ATTR    ((2 << GDRDRV_MAJOR_VERSION_SHIFT) | 5)
#define GDRDRV_MINIMUM_VERSION_WITH_PIN_BUFFER_V2 ((2 << GDRDRV_MAJOR_VERSION_SHIFT) | 5)
#define GDRDRV_MINIMUM_VERSION_WITH_REQ_MAPPING_TYPE ((2 << GDRDRV_MAJOR_VERSION_SHIFT) | 5)

#define GDRDRV_IOCTL                 0xDA

typedef enum {
    GDR_MR_NONE = 0,
    GDR_MR_WC = 1,
    GDR_MR_CACHING = 2,
    GDR_MR_DEVICE = 3
} gdr_mr_type_t;

typedef enum {
    GDRDRV_ATTR_USE_PERSISTENT_MAPPING = 1,
    GDRDRV_ATTR_SUPPORT_PIN_FLAG_FORCE_PCIE = 2
} gdrdrv_attr_t;

typedef __u64 gdr_hnd_t;

//-----------

struct GDRDRV_IOC_PIN_BUFFER_PARAMS
{
    // in
    __u64 addr;
    __u64 size;
    __u64 p2p_token;
    __u32 va_space;
    // out
    gdr_hnd_t handle;
};

#define GDRDRV_IOC_PIN_BUFFER _IOWR(GDRDRV_IOCTL, 1, struct GDRDRV_IOC_PIN_BUFFER_PARAMS)

//-----------

struct GDRDRV_IOC_UNPIN_BUFFER_PARAMS
{
    // in
    gdr_hnd_t handle;
};

#define GDRDRV_IOC_UNPIN_BUFFER _IOWR(GDRDRV_IOCTL, 2, struct GDRDRV_IOC_UNPIN_BUFFER_PARAMS *)

//-----------

struct GDRDRV_IOC_GET_CB_FLAG_PARAMS
{
    // in
    gdr_hnd_t handle;
    // out
    __u32 flag;
};

#define GDRDRV_IOC_GET_CB_FLAG _IOWR(GDRDRV_IOCTL, 3, struct GDRDRV_IOC_GET_CB_FLAG_PARAMS *)

//-----------

struct GDRDRV_IOC_GET_INFO_PARAMS
{
    // in
    gdr_hnd_t handle;
    // out
    __u64 va;
    __u64 mapped_size;
    __u32 page_size;
    __u32 tsc_khz;
    __u64 tm_cycles;
    __u32 mapped;
    __u32 wc_mapping;
    __u64 paddr;
};

#define GDRDRV_IOC_GET_INFO _IOWR(GDRDRV_IOCTL, 4, struct GDRDRV_IOC_GET_INFO_PARAMS *)

//-----------

struct GDRDRV_IOC_GET_INFO_V2_PARAMS
{
    // in
    gdr_hnd_t handle;
    // out
    __u64 va;
    __u64 mapped_size;
    __u32 page_size;
    __u32 tsc_khz;
    __u64 tm_cycles;
    __u32 mapping_type;
    __u64 paddr;
};

#define GDRDRV_IOC_GET_INFO_V2 _IOWR(GDRDRV_IOCTL, 5, struct GDRDRV_IOC_GET_INFO_V2_PARAMS *)

//-----------

struct GDRDRV_IOC_GET_ATTR_PARAMS
{
    // in
    __u32 attr; // gdrdrv_attr_t
    // out
    __u32 val;
};

#define GDRDRV_IOC_GET_ATTR _IOWR(GDRDRV_IOCTL, 6, struct GDRDRV_IOC_GET_ATTR_PARAMS *)

//-----------

struct GDRDRV_IOC_PIN_BUFFER_V2_PARAMS
{
    // in
    __u64 addr;
    __u64 size;
    __u32 flags;
    __u32 pad;
    // out
    gdr_hnd_t handle;
};

#define GDRDRV_PIN_BUFFER_FLAG_DEFAULT ((__u32)0x0)
// On supported platforms, force the creation of GPU BAR1 mappings
#define GDRDRV_PIN_BUFFER_FLAG_FORCE_PCIE ((__u32)0x1)

#define GDRDRV_IOC_PIN_BUFFER_V2 _IOWR(GDRDRV_IOCTL, 7, struct GDRDRV_IOC_PIN_BUFFER_V2_PARAMS)

//-----------

struct GDRDRV_IOC_REQ_MAPPING_TYPE_PARAMS
{
    // in
    gdr_hnd_t handle;
    __u32 mapping_type;
};

#define GDRDRV_IOC_REQ_MAPPING_TYPE _IOWR(GDRDRV_IOCTL, 8, struct GDRDRV_IOC_REQ_MAPPING_TYPE_PARAMS *)

//-----------

struct GDRDRV_IOC_GET_VERSION_PARAMS
{
    // out
    __u32 gdrdrv_version;
    __u32 minimum_gdr_api_version;
};

#define GDRDRV_IOC_GET_VERSION _IOWR(GDRDRV_IOCTL, 255, struct GDRDRV_IOC_GET_VERSION_PARAMS *)

//-----------

#endif // __GDR_DRV_H__
