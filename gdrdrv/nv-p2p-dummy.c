/*
 * Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
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

/*
 * Warning: this kernel module is only needed at compile time.
 *
 * Long story is that this module is here only to produce the correct
 * module versions related to the very kernel where the other module (the
 * interesting one) is going to be compiled.  In other words, this module
 * produce the same symbol versions as the real NVIDIA kernel-mode driver.
 *
 * Downside: the function signatures must be kept up-to-date.
 */

#include <linux/version.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/slab.h>
#include <linux/types.h>
#include <linux/delay.h>
#include <linux/compiler.h>
#include <linux/string.h>
#include <linux/uaccess.h>
#include <linux/fs.h>
#include <linux/list.h>
#include <linux/mm.h>
#include <linux/io.h>

#include "nv-p2p.h"

MODULE_AUTHOR("drossetti@nvidia.com");
MODULE_LICENSE("MIT");
MODULE_DESCRIPTION("P2P dummy kernel-mode driver");
MODULE_VERSION("1.0");

int nvidia_p2p_init_mapping(uint64_t p2p_token,
                            struct nvidia_p2p_params *params,
                            void (*destroy_callback)(void *data),
                            void *data)
{
    return -EINVAL;
}
EXPORT_SYMBOL(nvidia_p2p_init_mapping);

int nvidia_p2p_destroy_mapping(uint64_t p2p_token)
{
    return -EINVAL;
}
EXPORT_SYMBOL(nvidia_p2p_destroy_mapping);

int nvidia_p2p_get_pages(uint64_t p2p_token, uint32_t va_space,
                         uint64_t virtual_address,
                         uint64_t length,
                         struct nvidia_p2p_page_table **page_table,
                         void (*free_callback)(void *data),
                         void *data)
{
    return -EINVAL;
}
EXPORT_SYMBOL(nvidia_p2p_get_pages);

int nvidia_p2p_put_pages(uint64_t p2p_token, uint32_t va_space,
                         uint64_t virtual_address,
                         struct nvidia_p2p_page_table *page_table)
{
    return -EINVAL;
}
EXPORT_SYMBOL(nvidia_p2p_put_pages);

int nvidia_p2p_free_page_table(struct nvidia_p2p_page_table *page_table)
{
    return -EINVAL;
}
EXPORT_SYMBOL(nvidia_p2p_free_page_table);

static int __init nv_p2p_dummy_init(void)
{
    return 0;
}

static void __exit nv_p2p_dummy_cleanup(void)
{
}

module_init(nv_p2p_dummy_init);
module_exit(nv_p2p_dummy_cleanup);

/*
 * Local variables:
 *  c-indent-level: 4
 *  c-basic-offset: 4
 *  tab-width: 4
 *  indent-tabs-mode: nil
 * End:
 */
