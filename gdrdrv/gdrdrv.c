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
#include <linux/timex.h>
#include <linux/timer.h>

#ifndef random_get_entropy
#define random_get_entropy()	get_cycles()
#endif

#if defined(CONFIG_X86_64) || defined(CONFIG_X86_32)
static inline pgprot_t pgprot_modify_writecombine(pgprot_t old_prot)
{
    pgprot_t new_prot = old_prot;
    pgprot_val(new_prot) &= ~(_PAGE_PSE | _PAGE_PCD | _PAGE_PWT);
    new_prot = __pgprot(pgprot_val(new_prot) | _PAGE_PWT);
    return new_prot;
}
#define get_tsc_khz() cpu_khz // tsc_khz
#elif defined(CONFIG_PPC64)
static inline pgprot_t pgprot_modify_writecombine(pgprot_t old_prot)
{
    return pgprot_writecombine(old_prot);
}
#define get_tsc_khz() (get_cycles()/1000) // dirty hack
#else
#error "X86_64/32 or PPC64 is required"
#endif

#include "gdrdrv.h"
#include "nv-p2p.h"

//-----------------------------------------------------------------------------

#ifndef NVIDIA_P2P_MAJOR_VERSION_MASK
#define NVIDIA_P2P_MAJOR_VERSION_MASK   0xffff0000
#endif
#ifndef NVIDIA_P2P_MINOR_VERSION_MASK
#define NVIDIA_P2P_MINOR_VERSION_MASK   0x0000ffff
#endif

#ifndef NVIDIA_P2P_MAJOR_VERSION
#define NVIDIA_P2P_MAJOR_VERSION(v) \
    (((v) & NVIDIA_P2P_MAJOR_VERSION_MASK) >> 16)
#endif

#ifndef NVIDIA_P2P_MINOR_VERSION
#define NVIDIA_P2P_MINOR_VERSION(v) \
    (((v) & NVIDIA_P2P_MINOR_VERSION_MASK))
#endif

#ifndef NVIDIA_P2P_MAJOR_VERSION_MATCHES
#define NVIDIA_P2P_MAJOR_VERSION_MATCHES(p, v) \
    (NVIDIA_P2P_MAJOR_VERSION((p)->version) == NVIDIA_P2P_MAJOR_VERSION(v))
#endif

#ifndef NVIDIA_P2P_VERSION_COMPATIBLE
#define NVIDIA_P2P_VERSION_COMPATIBLE(p, v)             \
    (NVIDIA_P2P_MAJOR_VERSION_MATCHES(p, v) &&          \
    (NVIDIA_P2P_MINOR_VERSION((p)->version) >= NVIDIA_P2P_MINOR_VERSION(v)))
#endif

#ifndef NVIDIA_P2P_PAGE_TABLE_VERSION_COMPATIBLE
#define NVIDIA_P2P_PAGE_TABLE_VERSION_COMPATIBLE(p) \
    NVIDIA_P2P_VERSION_COMPATIBLE(p, NVIDIA_P2P_PAGE_TABLE_VERSION)
#endif

//-----------------------------------------------------------------------------

#define DEVNAME "gdrdrv"

#define gdr_msg(KRNLVL, FMT, ARGS...) printk(KRNLVL DEVNAME ":" FMT, ## ARGS)
//#define gdr_msg(KRNLVL, FMT, ARGS...) printk_ratelimited(KRNLVL DEVNAME ":" FMT, ## ARGS)

static int dbg_enabled = 0;
#define gdr_dbg(FMT, ARGS...)                               \
    do {                                                    \
        if (dbg_enabled)                                    \
            gdr_msg(KERN_DEBUG, FMT, ## ARGS);              \
    } while(0)

static int info_enabled = 0;
#define gdr_info(FMT, ARGS...)                               \
    do {                                                     \
        if (info_enabled)                                    \
            gdr_msg(KERN_INFO, FMT, ## ARGS);                \
    } while(0)

#define gdr_err(FMT, ARGS...)                               \
    gdr_msg(KERN_DEBUG, FMT, ## ARGS)

//-----------------------------------------------------------------------------

MODULE_AUTHOR("drossetti@nvidia.com");
MODULE_LICENSE("MIT");
MODULE_DESCRIPTION("GDRCopy kernel-mode driver");
MODULE_VERSION("1.1");
module_param(dbg_enabled, int, 0000);
MODULE_PARM_DESC(dbg_enabled, "enable debug tracing");
module_param(info_enabled, int, 0000);
MODULE_PARM_DESC(info_enabled, "enable info tracing");

//-----------------------------------------------------------------------------

#define GPU_PAGE_SHIFT   16
#define GPU_PAGE_SIZE    ((u64)1 << GPU_PAGE_SHIFT)
#define GPU_PAGE_OFFSET  (GPU_PAGE_SIZE-1)
#define GPU_PAGE_MASK    (~GPU_PAGE_OFFSET)

#ifndef MAX
#define MAX(a,b) ((a) > (b) ? a : b)
#endif

#ifndef MIN
#define MIN(a,b) ((a) < (b) ? a : b)
#endif


// compatibility with old Linux kernels

#ifndef ACCESS_ONCE
#define ACCESS_ONCE(x) (*(volatile typeof(x) *)&(x))
#endif

//-----------------------------------------------------------------------------

struct gdr_mr {
    struct list_head node;
    gdr_hnd_t handle;
    u64 offset;
    u64 length;
    u64 p2p_token;
    u32 va_space;
    u32 page_size;
    u64 va;
    u64 mapped_size;
    nvidia_p2p_page_table_t *page_table;
    int cb_flag;
    cycles_t tm_cycles;
    unsigned int tsc_khz;
};
typedef struct gdr_mr gdr_mr_t;


struct gdr_info {
    // simple low-performance linked-list implementation
    struct list_head mr_list;
    struct mutex lock;
};
typedef struct gdr_info gdr_info_t;

//-----------------------------------------------------------------------------

static int gdrdrv_major = 0;
//static gdr_mr_t *mr = NULL;

//-----------------------------------------------------------------------------

static int gdrdrv_open(struct inode *inode, struct file *filp)
{
    unsigned int minor = MINOR(inode->i_rdev);
    int ret = 0;
    gdr_info_t *info = NULL;

    gdr_dbg("minor=%d\n", minor);
    if(minor >= 1) {
        gdr_err("device minor number too big!\n");
        ret = -ENXIO;
        goto out;
    }

    info = kmalloc(sizeof(gdr_info_t), GFP_KERNEL);
    if (!info) {
        gdr_err("can't alloc kernel memory\n");
        ret = -ENOMEM;
        goto out;
    }

    INIT_LIST_HEAD(&info->mr_list);
    mutex_init(&info->lock);

    filp->private_data = info;

out:
    return ret;
}

//-----------------------------------------------------------------------------

static int gdrdrv_release(struct inode *inode, struct file *filp)
{
    int retcode;
    gdr_info_t *info = filp->private_data;
    gdr_mr_t *mr = NULL;
    struct list_head *p, *n;

    gdr_dbg("closing\n");

    // BUG: do proper locking here
    list_for_each_safe(p, n, &info->mr_list) {
        mr = list_entry(p, gdr_mr_t, node);
        gdr_info("freeing MR=%p\n", mr);
        if (!ACCESS_ONCE(mr->cb_flag)) {
            retcode = nvidia_p2p_put_pages(mr->p2p_token, mr->va_space, mr->va, mr->page_table);
            if (retcode) {
                gdr_err("error while calling put_pages\n");
            }
        }
        mutex_lock(&info->lock);
        list_del(&mr->node);
        mutex_unlock(&info->lock);
        memset(mr, 0, sizeof(*mr));
        kfree(mr);
    }
    kfree(info);
    filp->private_data = NULL;

    return 0;
}

//-----------------------------------------------------------------------------

static gdr_mr_t *gdr_mr_from_handle(gdr_info_t *info, gdr_hnd_t handle)
{
    gdr_mr_t *mr = NULL;
    struct list_head *p;

    list_for_each(p, &info->mr_list) {
        mr = list_entry(p, gdr_mr_t, node);
        if (handle == mr->handle)
            break;
    }

    return mr;
}

//-----------------------------------------------------------------------------
// off is page aligned, because of the kernel interface
// could abuse lower bits for other purposes

static gdr_hnd_t gdrdrv_handle_from_off(unsigned long off)
{
    return (gdr_hnd_t)(off);
}

//-----------------------------------------------------------------------------
// BUG: mr access is not explicitly protected by a lock

static void gdrdrv_get_pages_free_callback(void *data)
{
    gdr_mr_t *mr = data;
    nvidia_p2p_page_table_t *page_table = NULL;
    gdr_info("free callback\n");
    // DR: can't take the info->lock here due to potential AB-BA
    // deadlock with internal NV driver lock(s)
    ACCESS_ONCE(mr->cb_flag) = 1;
    wmb();
    page_table = xchg(&mr->page_table, NULL);
    if (page_table) {
        nvidia_p2p_free_page_table(page_table);
        barrier();
    } else {
        gdr_err("ERROR: free callback, page_table is NULL\n");
    }
}

//-----------------------------------------------------------------------------

static int gdrdrv_pin_buffer(gdr_info_t *info, void __user *_params)
{
    struct GDRDRV_IOC_PIN_BUFFER_PARAMS params = {0};
    int ret = 0;
    struct nvidia_p2p_page_table *page_table = NULL;
    u64 page_virt_start;
    u64 page_virt_end;
    size_t rounded_size;
    gdr_mr_t *mr = NULL;
    cycles_t ta, tb;

    if (copy_from_user(&params, _params, sizeof(params))) {
        gdr_err("copy_from_user failed on user pointer %p\n", _params);
        ret = -EFAULT;
        goto out;
    }

    if (!params.addr) {
        gdr_err("NULL device pointer\n");
        ret = -EINVAL;
        goto out;
    }

    mr = kmalloc(sizeof(gdr_mr_t), GFP_KERNEL);
    if (!mr) {
        gdr_err("can't alloc kernel memory\n");
        ret = -ENOMEM;
        goto out;
    }
    memset(mr, 0, sizeof(*mr));

    // do proper alignment, as required by RM
    page_virt_start  = params.addr & GPU_PAGE_MASK;
    //page_virt_end    = (params.addr + params.size + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK;
    page_virt_end    = params.addr + params.size - 1;
    rounded_size     = page_virt_end - page_virt_start + 1;
    //rounded_size     = (params.addr & GPU_PAGE_OFFSET) + params.size;

    mr->offset       = params.addr & GPU_PAGE_OFFSET;
    mr->length       = params.size;
    mr->p2p_token    = params.p2p_token;
    mr->va_space     = params.va_space;
    mr->va           = page_virt_start;
    mr->mapped_size  = rounded_size;
    mr->page_table   = NULL;
    mr->cb_flag      = 0;
    mr->handle       = random_get_entropy() & GDR_HANDLE_MASK; // this is a hack, we need something really unique and randomized

    gdr_info("invoking nvidia_p2p_get_pages(va=0x%llx len=%lld p2p_tok=%llx va_tok=%x)\n",
             mr->va, mr->mapped_size, mr->p2p_token, mr->va_space);

    ta = get_cycles();
    ret = nvidia_p2p_get_pages(mr->p2p_token, mr->va_space, mr->va, mr->mapped_size, &page_table,
                               gdrdrv_get_pages_free_callback, mr);
    tb = get_cycles();
    if (ret < 0) {
        gdr_err("nvidia_p2p_get_pages(va=%llx len=%lld p2p_token=%llx va_space=%x) failed [ret = %d]\n",
                mr->va, mr->mapped_size, mr->p2p_token, mr->va_space, ret);
        goto out;
    }
    mr->page_table = page_table;
    mr->tm_cycles = tb - ta;
    mr->tsc_khz = get_tsc_khz();

    // check version before accessing page table
    if (!NVIDIA_P2P_PAGE_TABLE_VERSION_COMPATIBLE(page_table)) {
        gdr_err("incompatible page table version 0x%08x\n", page_table->version);
        goto out;
    }

    switch(page_table->page_size) {
    case NVIDIA_P2P_PAGE_SIZE_4KB:
        mr->page_size = 4*1024;
        break;
    case NVIDIA_P2P_PAGE_SIZE_64KB:
        mr->page_size = 64*1024;
        break;
    case NVIDIA_P2P_PAGE_SIZE_128KB:
        mr->page_size = 128*1024;
        break;
    default:
        gdr_err("unexpected page_size\n");
        ret = -EINVAL;
        goto out;
    }

    // we are not really ready for a different page size
    if(page_table->page_size != NVIDIA_P2P_PAGE_SIZE_64KB) {
        gdr_err("nvidia_p2p_get_pages assumption of 64KB pages failed size_id=%d\n", page_table->page_size);
        ret = -EINVAL;
        goto out;
    }
    {
        int i;
        gdr_dbg("page table entries: %d\n", page_table->entries);
        for (i=0; i<page_table->entries; ++i) {
            gdr_dbg("page[%d]=0x%016llx\n", i, page_table->pages[i]->physical_address);
        }
    }


    // here a typical driver would use the page_table to fill in some HW
    // DMA data structure

    params.handle = mr->handle;

    mutex_lock(&info->lock);
    list_add(&mr->node, &info->mr_list);
    mutex_unlock(&info->lock);

out:

    if (ret && mr && mr->page_table) {
        gdr_err("error, calling p2p_put_pages\n");
        nvidia_p2p_put_pages(mr->p2p_token, mr->va_space, mr->va, mr->page_table);
        page_table = NULL;
        mr->page_table = NULL;
    }

    if (ret && mr) {
        kfree(mr);
        memset(mr, 0, sizeof(*mr));
        mr = NULL;
    }

    if (!ret && copy_to_user(_params, &params, sizeof(params))) {
        gdr_err("copy_to_user failed on user pointer %p\n", _params);
        ret = -EFAULT;
    }

    return ret;
}

//-----------------------------------------------------------------------------

static int gdrdrv_unpin_buffer(gdr_info_t *info, void __user *_params)
{
    struct GDRDRV_IOC_UNPIN_BUFFER_PARAMS params = {0};
    int ret = 0, retcode = 0;
    gdr_mr_t *mr = NULL;

    if (copy_from_user(&params, _params, sizeof(params))) {
        gdr_err("copy_from_user failed on user pointer %p\n", _params);
        return -EFAULT;
    }

    mr = gdr_mr_from_handle(info, params.handle);
    if (NULL == mr) {
        gdr_err("unexpected handle %x while unmapping buffer\n", params.handle);
        return -EINVAL;
    }

    if (!mr->cb_flag) {
        gdr_info("invoking nvidia_p2p_put_pages(va=0x%llx p2p_tok=%llx va_tok=%x)\n",
                 mr->va, mr->p2p_token, mr->va_space);
        retcode = nvidia_p2p_put_pages(mr->p2p_token, mr->va_space, mr->va, mr->page_table);
        if (retcode) {
            gdr_err("nvidia_p2p_put_pages error %d, async callback may have been fired\n", retcode);
        }
    } else {
        gdr_err("invoking unpin_buffer while callback has already been fired\n");
        // not returning an error here because further clean-up is
        // needed anyway
    }
    mutex_lock(&info->lock);
    list_del(&mr->node);
    mutex_unlock(&info->lock);
    memset(mr, 0, sizeof(*mr));
    kfree(mr);

    return ret;
}

//-----------------------------------------------------------------------------

static int gdrdrv_get_cb_flag(gdr_info_t *info, void __user *_params)
{
    struct GDRDRV_IOC_GET_CB_FLAG_PARAMS params = {0};
    int ret = 0;
    gdr_mr_t *mr = NULL;

    if (copy_from_user(&params, _params, sizeof(params))) {
        gdr_err("copy_from_user failed on user pointer %p\n", _params);
        return -EFAULT;
    }

    mr = gdr_mr_from_handle(info, params.handle);
    if (NULL == mr) {
        gdr_err("unexpected handle %x in get_cb_flag\n", params.handle);
        return -EINVAL;
    }

    params.flag = !!mr->cb_flag;

    if (copy_to_user(_params, &params, sizeof(params))) {
        gdr_err("copy_to_user failed on user pointer %p\n", _params);
        ret = -EFAULT;
    }

    return ret;
}

//-----------------------------------------------------------------------------

static int gdrdrv_get_info(gdr_info_t *info, void __user *_params)
{
    struct GDRDRV_IOC_GET_INFO_PARAMS params = {0};
    int ret = 0;
    gdr_mr_t *mr = NULL;

    if (copy_from_user(&params, _params, sizeof(params))) {
        gdr_err("copy_from_user failed on user pointer %p\n", _params);
        return -EFAULT;
    }

    mr = gdr_mr_from_handle(info, params.handle);
    if (NULL == mr) {
        gdr_err("unexpected handle %x in get_cb_flag\n", params.handle);
        return -EINVAL;
    }

    params.va          = mr->va;
    params.mapped_size = mr->mapped_size;
    params.page_size   = mr->page_size;
    params.tm_cycles   = mr->tm_cycles;
    params.tsc_khz     = mr->tsc_khz;

    if (copy_to_user(_params, &params, sizeof(params))) {
        gdr_err("copy_to_user failed on user pointer %p\n", _params);
        ret = -EFAULT;
    }

    return ret;
}

//-----------------------------------------------------------------------------

static int gdrdrv_ioctl(struct inode *inode, struct file *filp, unsigned int cmd, unsigned long arg)
{
    int ret = 0;
    gdr_info_t *info = filp->private_data;
    void __user *argp = (void __user *)arg;

    gdr_dbg("ioctl called (cmd 0x%x)\n", cmd);

    if (_IOC_TYPE(cmd) != GDRDRV_IOCTL) {
        gdr_err("malformed IOCTL code type=%08x\n", _IOC_TYPE(cmd));
        return -EINVAL;
    }

    switch (cmd) {
    case GDRDRV_IOC_PIN_BUFFER:
        ret = gdrdrv_pin_buffer(info, argp);
        break;

    case GDRDRV_IOC_UNPIN_BUFFER:
        ret = gdrdrv_unpin_buffer(info, argp);
        break;

    case GDRDRV_IOC_GET_CB_FLAG:
        ret = gdrdrv_get_cb_flag(info, argp);
        break;

    case GDRDRV_IOC_GET_INFO:
        ret = gdrdrv_get_info(info, argp);
        break;

    default:
        gdr_err("unsupported IOCTL code\n");
        ret = -ENOTTY;
    }
    return ret;
}

//-----------------------------------------------------------------------------

#ifdef HAVE_UNLOCKED_IOCTL
static long gdrdrv_unlocked_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
    return gdrdrv_ioctl(0, filp, cmd, arg);
}
#endif

/*----------------------------------------------------------------------------*/

static int gdrdrv_mmap_phys_mem_wcomb(struct vm_area_struct *vma, unsigned long vaddr, unsigned long paddr, size_t size)
{
    int ret = 0;
    unsigned long pfn;

    gdr_dbg("mmaping phys mem addr=0x%lx size=%zu at user virt addr=0x%lx\n", 
             paddr, size, vaddr);

    // in case the original user address was not properly host page-aligned
    if (0 != (paddr & (PAGE_SIZE-1))) {
        gdr_err("paddr=%lx, original mr address was not host page-aligned\n", paddr);
        ret = -EINVAL;
        goto out;
    }
    if (0 != (vaddr & (PAGE_SIZE-1))) {
        gdr_err("vaddr=%lx, trying to map to non page-aligned vaddr\n", vaddr);
        ret = -EINVAL;
        goto out;
    }

    pfn = paddr >> PAGE_SHIFT;
    gdr_dbg("pfn=0x%lx\n", pfn);

#if LINUX_VERSION_CODE <= KERNEL_VERSION(2,6,9)
    vma->vm_pgoff = pfn;
    vma->vm_flags |= VM_RESERVED;
    vma->vm_flags |= VM_IO;
    if (remap_page_range(vma, vaddr, paddr, size, vma->vm_page_prot)) {
        gdr_err("error in remap_page_range()\n");
        ret = -EAGAIN;
        goto out;
    }
#else
    vma->vm_page_prot = pgprot_modify_writecombine(vma->vm_page_prot);
    gdr_dbg("calling io_remap_pfn_range() vma=%p vaddr=%lx pfn=%lx size=%zu\n", 
            vma, vaddr, pfn, size);
    if (io_remap_pfn_range(vma, vaddr, pfn, size, vma->vm_page_prot)) {
        gdr_err("error in remap_pfn_range()\n");
        ret = -EAGAIN;
        goto out;
    }
#endif

out:
    return ret;
}

//-----------------------------------------------------------------------------
// BUG: should obtain GPU_PAGE_SIZE from page_table!!!

static int gdrdrv_mmap(struct file *filp, struct vm_area_struct *vma)
{
    int ret = 0;
    size_t size = vma->vm_end - vma->vm_start;
    gdr_info_t* info = filp->private_data;
    gdr_hnd_t handle;
    gdr_mr_t *mr = NULL;
    u64 offset;
    int p = 0;
    unsigned long vaddr, prev_page_paddr;
    int phys_contiguous = 1;

    gdr_info("mmap start=0x%lx size=%zu off=0x%lx\n", vma->vm_start, size, vma->vm_pgoff);

    handle = gdrdrv_handle_from_off(vma->vm_pgoff);
    mr = gdr_mr_from_handle(info, handle);
    if (!mr) {
        ret = -EINVAL;
        goto out;
    }
    offset = mr->offset;
    if (mr->cb_flag) {
        gdr_dbg("mr has been invalidated\n");
        ret = -EINVAL;
        goto out;
    }
    if (!mr->page_table) {
        gdr_dbg("invalid mr state\n");
        ret = -EINVAL;
        goto out;
    }
    if (mr->page_table->entries <= 0) {
        gdr_dbg("invalid entries in page table\n");
        ret = -EINVAL;
        goto out;
    }
    if (offset) {
        gdr_dbg("offset != 0 is not supported\n");
        ret = -EINVAL;
        goto out;
    }    

    p = 0;
    vaddr = vma->vm_start;
    prev_page_paddr = mr->page_table->pages[0]->physical_address;
    phys_contiguous = 1;
    for(p = 1; p < mr->page_table->entries; ++p) {
        struct nvidia_p2p_page *page = mr->page_table->pages[p];
        unsigned long page_paddr = page->physical_address;
        if (prev_page_paddr + GPU_PAGE_SIZE != page_paddr) {
            gdr_dbg("page table entry %d is non-contiguous with previous\n", p);
            phys_contiguous = 0;
            break;
        }
        prev_page_paddr = page_paddr;
    }

    if (phys_contiguous) {
        // offset not supported
        size_t len = GPU_PAGE_SIZE * mr->page_table->entries;
        unsigned long page0_paddr = mr->page_table->pages[0]->physical_address;
        gdr_dbg("offset=%llx len=%zu vaddr+offset=%llx paddr+offset=%llx\n", 
                offset, len, vaddr+offset, page0_paddr + offset);
        ret = gdrdrv_mmap_phys_mem_wcomb(vma, 
                                         vaddr + offset, 
                                         page0_paddr + offset, 
                                         len);
        if (ret) {
            gdr_err("mmap error\n");
            goto out;
        }
    } else {
        if (offset > GPU_PAGE_SIZE) {
            gdr_dbg("offset > GPU_PAGE_SIZE-offset is not supported\n");
            ret = -EINVAL;
            goto out;
        }    

        // If not contiguous, map individual GPU pages separately.
        // In this case, write-combining performance can be really bad, not 
        // sure why.
        while(size && p < mr->page_table->entries) {
            struct nvidia_p2p_page *page = mr->page_table->pages[p];
            unsigned long page_paddr = page->physical_address;
            size_t len = MIN(GPU_PAGE_SIZE-offset, size);

            gdr_dbg("mapping page_i=%d offset=%llx len=%zu vaddr=%lx\n", 
                    p, offset, len, vaddr);

            if (offset > GPU_PAGE_SIZE) {
                gdr_dbg("skipping a whole GPU page\n");
                ++p;
                offset -= GPU_PAGE_SIZE;
                vaddr += GPU_PAGE_SIZE;
                continue;
            }

            ret = gdrdrv_mmap_phys_mem_wcomb(vma, 
                                             vaddr, 
                                             page_paddr + offset, 
                                             len);
            if (ret) {
                gdr_err("mmap error\n");
                goto out;
            }

            vaddr += len;
            size -= len;
            offset = 0;
            ++p;
        }
    }

out:
    // TBD: don't leave partial mappings on error

    return ret;
}

//-----------------------------------------------------------------------------

struct file_operations gdrdrv_fops = {
    .owner    = THIS_MODULE,

#ifdef HAVE_UNLOCKED_IOCTL
    .unlocked_ioctl = gdrdrv_unlocked_ioctl,
#else
    .ioctl    = gdrdrv_ioctl,
#endif
    .open     = gdrdrv_open,
    .release  = gdrdrv_release,
    .mmap     = gdrdrv_mmap
};

//-----------------------------------------------------------------------------

static int __init gdrdrv_init(void)
{
    int result;

    result = register_chrdev(gdrdrv_major, DEVNAME, &gdrdrv_fops);
    if (result < 0) {
        gdr_err("can't get major %d\n", gdrdrv_major);
        return result;
    }
    if (gdrdrv_major == 0) gdrdrv_major = result; /* dynamic */

    gdr_msg(KERN_INFO, "device registered with major number %d\n", gdrdrv_major);
    gdr_msg(KERN_INFO, "dbg traces %s, info traces %s", dbg_enabled ? "enabled" : "disabled", info_enabled ? "enabled" : "disabled");

    //gdrdrv_init_devices();/* fills to zero the device array */

    return 0;
}

//-----------------------------------------------------------------------------

static void __exit gdrdrv_cleanup(void)
{
    gdr_msg(KERN_INFO, "unregistering major number %d\n", gdrdrv_major);

    /* cleanup_module is never called if registering failed */
    unregister_chrdev(gdrdrv_major, DEVNAME);
}

//-----------------------------------------------------------------------------

module_init(gdrdrv_init);
module_exit(gdrdrv_cleanup);

/*
 * Local variables:
 *  c-indent-level: 4
 *  c-basic-offset: 4
 *  tab-width: 4
 *  indent-tabs-mode: nil
 * End:
 */
