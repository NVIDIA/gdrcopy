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
#include <linux/sched.h>

#if LINUX_VERSION_CODE <= KERNEL_VERSION(2,6,32)
/**
 * This API is available after Linux kernel 2.6.32
 */
void address_space_init_once(struct address_space *mapping)
{
    memset(mapping, 0, sizeof(*mapping));
    INIT_RADIX_TREE(&mapping->page_tree, GFP_ATOMIC);

#if LINUX_VERSION_CODE <= KERNEL_VERSION(2,6,26)
    //  
    // The .tree_lock member variable was changed from type rwlock_t, to
    // spinlock_t, on 25 July 2008, by mainline commit
    // 19fd6231279be3c3bdd02ed99f9b0eb195978064.
    //  
    rwlock_init(&mapping->tree_lock);
#else
    spin_lock_init(&mapping->tree_lock);
#endif

    spin_lock_init(&mapping->i_mmap_lock);
    INIT_LIST_HEAD(&mapping->private_list);
    spin_lock_init(&mapping->private_lock);
    INIT_RAW_PRIO_TREE_ROOT(&mapping->i_mmap);
    INIT_LIST_HEAD(&mapping->i_mmap_nonlinear);
}
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
static inline int gdr_pfn_is_ram(unsigned long pfn)
{
    // page_is_ram is GPL-only. Regardless there are no x86_64
    // platforms supporting coherent GPU mappings, so we would not use
    // this function anyway.
    return 0;
}

#elif defined(CONFIG_PPC64)
#include <asm/reg.h>
static inline pgprot_t pgprot_modify_writecombine(pgprot_t old_prot)
{
    return pgprot_writecombine(old_prot);
}
#define get_tsc_khz() (get_cycles()/1000) // dirty hack
static inline int gdr_pfn_is_ram(unsigned long pfn)
{
    // catch platforms, e.g. POWER8, POWER9 with GPUs not attached via NVLink,
    // where GPU memory is non-coherent
#if 0
    // unfortunately this module is MIT, and page_is_ram is GPL-only.
    return page_is_ram(pfn);
#else
    unsigned long start = pfn << PAGE_SHIFT;
    unsigned long mask_47bits = (1UL<<47)-1;
    return 0 == (start & ~mask_47bits);
#endif
}

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
    enum { GDR_MR_NONE, GDR_MR_WC, GDR_MR_CACHING } cpu_mapping_type;
    nvidia_p2p_page_table_t *page_table;
    int cb_flag;
    cycles_t tm_cycles;
    unsigned int tsc_khz;
    struct vm_area_struct *vma;
    struct address_space *mapping;
};
typedef struct gdr_mr gdr_mr_t;

static int gdr_mr_is_mapped(gdr_mr_t *mr)
{
    return mr->cpu_mapping_type != GDR_MR_NONE;
}

static int gdr_mr_is_wc_mapping(gdr_mr_t *mr)
{
    return (mr->cpu_mapping_type == GDR_MR_WC) ? 1 : 0;
}

static inline void gdrdrv_zap_vma(struct address_space *mapping, struct vm_area_struct *vma)
{
    // This function is mainly used for files and the address is relative to
    // the file offset. We use vma->pg_off here to unmap this entire range but
    // not the other mapped ranges.
    unmap_mapping_range(mapping, vma->vm_pgoff << PAGE_SHIFT, vma->vm_end - vma->vm_start, 0);
}

static void gdr_mr_destroy_all_mappings(gdr_mr_t *mr)
{
    // there is a single mapping at the moment
    if (mr->vma)
        gdrdrv_zap_vma(mr->mapping, mr->vma);
}

//-----------------------------------------------------------------------------

struct gdr_info {
    // simple low-performance linked-list implementation
    struct list_head        mr_list;
    struct mutex            lock;

    // Pointer to the pid struct of the creator process. We do not use
    // numerical pid here to avoid issues from pid reuse.
    struct pid             *pid;

    // Address space uniqued to this opened file. We need to create a new one
    // because filp->f_mapping usually points to inode->i_mapping.
    struct address_space    mapping;

    // The handle number and mmap's offset are equivalent. However, the mmap
    // offset is used by the linux kernel when doing m(un)map; hence the range
    // cannot be overlapped. We place two ranges next two each other to avoid
    // this issue.
    gdr_hnd_t               next_handle;
    int                     next_handle_overflow;
};
typedef struct gdr_info gdr_info_t;

//-----------------------------------------------------------------------------

static int gdrdrv_major = 0;
static int gdrdrv_cpu_can_cache_gpu_mappings = 0;

//-----------------------------------------------------------------------------

static int gdrdrv_open(struct inode *inode, struct file *filp)
{
    unsigned int minor = MINOR(inode->i_rdev);
    int ret = 0;
    gdr_info_t *info = NULL;

    gdr_dbg("minor=%d filep=0x%px\n", minor, filp);
    if(minor >= 1) {
        gdr_err("device minor number too big!\n");
        ret = -ENXIO;
        goto out;
    }

    info = kzalloc(sizeof(gdr_info_t), GFP_KERNEL);
    if (!info) {
        gdr_err("can't alloc kernel memory\n");
        ret = -ENOMEM;
        goto out;
    }

    INIT_LIST_HEAD(&info->mr_list);
    mutex_init(&info->lock);

    // GPU driver does not support sharing GPU allocations at fork time. Hence
    // here we track the process owning the driver fd and prevent other process
    // to use it.
    info->pid = task_pid(current);

    address_space_init_once(&info->mapping);
    info->mapping.host = inode;
    info->mapping.a_ops = inode->i_mapping->a_ops;
#if LINUX_VERSION_CODE < KERNEL_VERSION(4,0,0)
    info->mapping.backing_dev_info = inode->i_mapping->backing_dev_info;
#endif
    filp->f_mapping = &info->mapping;

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

    if (!info) {
        gdr_err("filp contains no info\n");
        return -EIO;
    }
    // Check that the caller is the same process that did gdrdrv_open
    if (info->pid != task_pid(current)) {
        gdr_err("filp is not opened by the current process\n");
        return -EACCES;
    }

    mutex_lock(&info->lock);
    list_for_each_safe(p, n, &info->mr_list) {
        mr = list_entry(p, gdr_mr_t, node);
        gdr_info("freeing MR=0x%px\n", mr);
        if (gdr_mr_is_mapped(mr)) {
            mutex_unlock(&info->lock);
            gdr_mr_destroy_all_mappings(mr);
            mutex_lock(&info->lock);
        }
        if (!ACCESS_ONCE(mr->cb_flag)) {
            mutex_unlock(&info->lock);
            // this may call the invalidation cb, e.g. on L4T
            retcode = nvidia_p2p_put_pages(mr->p2p_token, mr->va_space, mr->va, mr->page_table);
            if (retcode) {
                gdr_err("error while calling put_pages\n");
            }
            mutex_lock(&info->lock);
        }
        list_del(&mr->node);
        //memset(mr, 0, sizeof(*mr));
        kzfree(mr);
    }
    mutex_unlock(&info->lock);

    filp->f_mapping = NULL;

    kfree(info);
    filp->private_data = NULL;

    return 0;
}

//-----------------------------------------------------------------------------

static gdr_mr_t *gdr_mr_from_handle_unlocked(gdr_info_t *info, gdr_hnd_t handle)
{
    gdr_mr_t *mr = NULL;
    struct list_head *p;

    list_for_each(p, &info->mr_list) {
        mr = list_entry(p, gdr_mr_t, node);
        gdr_dbg("mr->handle=0x%llx handle=0x%llx\n", mr->handle, handle);
        if (handle == mr->handle)
            break;
    }

    return mr;
}

static gdr_mr_t *gdr_mr_from_handle(gdr_info_t *info, gdr_hnd_t handle)
{
    gdr_mr_t *mr;
    mutex_lock(&info->lock);
    mr = gdr_mr_from_handle_unlocked(info, handle);
    mutex_unlock(&info->lock);
    return mr;
}

//-----------------------------------------------------------------------------
// off is host page aligned, because of the kernel interface
// could abuse extra available bits for other purposes

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
    // can't take the info->lock here due to potential AB-BA
    // deadlock with internal NV driver lock(s)
    ACCESS_ONCE(mr->cb_flag) = 1;
    smp_wmb();
    page_table = xchg(&mr->page_table, NULL);
    if (page_table) {
        nvidia_p2p_free_page_table(page_table);
        if (gdr_mr_is_mapped(mr))
            gdr_mr_destroy_all_mappings(mr);
    } else {
        gdr_err("ERROR: free callback, page_table is NULL\n");
    }
}

//-----------------------------------------------------------------------------

/**
 * Generate mr->handle. This function should be called under info->lock.
 *
 * Prerequisite:
 * - mr->mapped_size is set and round to max(PAGE_SIZE, GPU_PAGE_SIZE)
 *
 * Return 0 if success, -1 if failed.
 */
static inline int gdr_generate_mr_handle(gdr_info_t *info, gdr_mr_t *mr)
{
    // The user-space library passes the memory (handle << PAGE_SHIFT) as the
    // mmap offset, and offsets are used to determine the VMAs to delete during
    // invalidation.  
    // Hence, we need [(handle << PAGE_SHIFT), (handle << PAGE_SHIFT) + size - 1] 
    // to correspond to a unique VMA.  Note that size here must match the
    // original mmap size

    gdr_hnd_t next_handle;

    WARN_ON(!mutex_is_locked(&info->lock));

    // We run out of handle, so fail.
    if (unlikely(info->next_handle_overflow))
        return -1;
    
    next_handle = info->next_handle + (mr->mapped_size >> PAGE_SHIFT);

    // The next handle will be overflowed, so we mark it.
    if (unlikely((next_handle & ((gdr_hnd_t)(-1) >> PAGE_SHIFT)) < info->next_handle))
        info->next_handle_overflow = 1;

    mr->handle = info->next_handle;
    info->next_handle = next_handle;

    return 0;
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
        gdr_err("copy_from_user failed on user pointer 0x%px\n", _params);
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
    mr->cpu_mapping_type = GDR_MR_NONE;
    mr->page_table   = NULL;
    mr->cb_flag      = 0;

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

    switch (page_table->page_size) {
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
    if (page_table->page_size != NVIDIA_P2P_PAGE_SIZE_64KB) {
        gdr_err("nvidia_p2p_get_pages assumption of 64KB pages failed size_id=%d\n", page_table->page_size);
        ret = -EINVAL;
        goto out;
    }
    {
        int i;
        gdr_dbg("page table entries: %d\n", page_table->entries);
        for (i=0; i<MIN(20,page_table->entries); ++i) {
            gdr_dbg("page[%d]=0x%016llx%s\n", i, page_table->pages[i]->physical_address, (i>19)?"and counting":"");
        }
    }

    // here a typical driver would use the page_table to fill in some HW
    // DMA data structure

    mutex_lock(&info->lock);
    if (gdr_generate_mr_handle(info, mr) != 0) {
        gdr_err("No address space left for BAR1 mapping.\n");
        ret = -ENOMEM;
    }

    if (!ret)
        list_add(&mr->node, &info->mr_list);
    mutex_unlock(&info->lock);

    params.handle = mr->handle;

out:

    if (ret && mr && mr->page_table) {
        gdr_err("error, calling p2p_put_pages\n");
        nvidia_p2p_put_pages(mr->p2p_token, mr->va_space, mr->va, mr->page_table);
        page_table = NULL;
        mr->page_table = NULL;
    }

    if (ret && mr) {
        memset(mr, 0, sizeof(*mr));
        kfree(mr);
        mr = NULL;
    }

    if (!ret && copy_to_user(_params, &params, sizeof(params))) {
        gdr_err("copy_to_user failed on user pointer 0x%px\n", _params);
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
        gdr_err("copy_from_user failed on user pointer 0x%px\n", _params);
        return -EFAULT;
    }

    // someone might try to traverse the list and/or to do something
    // to the mr at the same time, so let's lock here
    mutex_lock(&info->lock);
    mr = gdr_mr_from_handle_unlocked(info, params.handle);
    if (NULL == mr) {
        gdr_err("unexpected handle %llx while unmapping buffer\n", params.handle);
        ret = -EINVAL;
    } else {
        if (gdr_mr_is_mapped(mr)) {
            gdr_mr_destroy_all_mappings(mr);
        }
        list_del(&mr->node);
    }
    mutex_unlock(&info->lock);
    if (ret)
        goto out;
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
    //memset(mr, 0, sizeof(*mr));
    kzfree(mr);
 out:
    return ret;
}

//-----------------------------------------------------------------------------

static int gdrdrv_get_cb_flag(gdr_info_t *info, void __user *_params)
{
    struct GDRDRV_IOC_GET_CB_FLAG_PARAMS params = {0};
    int ret = 0;
    gdr_mr_t *mr = NULL;

    if (copy_from_user(&params, _params, sizeof(params))) {
        gdr_err("copy_from_user failed on user pointer 0x%px\n", _params);
        return -EFAULT;
    }
    mr = gdr_mr_from_handle(info, params.handle);
    if (NULL == mr) {
        gdr_err("unexpected handle %llx in get_cb_flag\n", params.handle);
        ret = -EINVAL;
        goto out;
    }

    params.flag = !!mr->cb_flag;

    if (copy_to_user(_params, &params, sizeof(params))) {
        gdr_err("copy_to_user failed on user pointer 0x%px\n", _params);
        ret = -EFAULT;
    }
 out:
    return ret;
}

//-----------------------------------------------------------------------------

static int gdrdrv_get_info(gdr_info_t *info, void __user *_params)
{
    struct GDRDRV_IOC_GET_INFO_PARAMS params = {0};
    int ret = 0;
    gdr_mr_t *mr = NULL;

    if (copy_from_user(&params, _params, sizeof(params))) {
        gdr_err("copy_from_user failed on user pointer 0x%px\n", _params);
        ret = -EFAULT;
        goto out;
    }

    mr = gdr_mr_from_handle(info, params.handle);
    if (NULL == mr) {
        gdr_err("unexpected handle %llx in get_cb_flag\n", params.handle);
        ret = -EINVAL;
        goto out;
    }

    params.va          = mr->va;
    params.mapped_size = mr->mapped_size;
    params.page_size   = mr->page_size;
    params.tm_cycles   = mr->tm_cycles;
    params.tsc_khz     = mr->tsc_khz;
    params.mapped      = gdr_mr_is_mapped(mr);
    params.wc_mapping  = gdr_mr_is_wc_mapping(mr);
    if (copy_to_user(_params, &params, sizeof(params))) {
        gdr_err("copy_to_user failed on user pointer 0x%px\n", _params);
        ret = -EFAULT;
    }
 out:
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

    if (!info) {
        gdr_err("filp contains no info\n");
        return -EIO;
    }
    // Check that the caller is the same process that did gdrdrv_open
    if (info->pid != task_pid(current)) {
        gdr_err("filp is not opened by the current process\n");
        return -EACCES;
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

void gdrdrv_vma_close(struct vm_area_struct *vma)
{
    gdr_mr_t *mr = (gdr_mr_t *)vma->vm_private_data;
    gdr_dbg("closing vma=0x%px vm_file=0x%px vm_private_data=0x%px mr=0x%px mr->vma=0x%px\n", vma, vma->vm_file, vma->vm_private_data, mr, mr->vma);
    // TODO: handle multiple vma's
    mr->vma = NULL;
    mr->cpu_mapping_type = GDR_MR_NONE;
}

/*----------------------------------------------------------------------------*/

static const struct vm_operations_struct gdrdrv_vm_ops = {
    .close = gdrdrv_vma_close,
};

/*----------------------------------------------------------------------------*/

static int gdrdrv_remap_gpu_mem(struct vm_area_struct *vma, unsigned long vaddr, unsigned long paddr, size_t size, int is_wcomb)
{
    int ret = 0;
    unsigned long pfn;

    gdr_dbg("mmaping phys mem addr=0x%lx size=%zu at user virt addr=0x%lx\n", 
             paddr, size, vaddr);

    if (!size) {
        gdr_dbg("size == 0\n");
        goto out;
    }
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

    // Disallow mmapped VMA to propagate to child processes
    vma->vm_flags |= VM_DONTCOPY;

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
    if (is_wcomb) {
        // override prot to create non-coherent WC mappings
        vma->vm_page_prot = pgprot_modify_writecombine(vma->vm_page_prot);
    } else {
        // by default, vm_page_prot should be set to create cached mappings
    }
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
    unsigned long vaddr;

    gdr_info("mmap filp=0x%px vma=0x%px vm_file=0x%px start=0x%lx size=%zu off=0x%lx\n", filp, vma, vma->vm_file, vma->vm_start, size, vma->vm_pgoff);

    if (!info) {
        gdr_err("filp contains no info\n");
        return -EIO;
    }
    // Check that the caller is the same process that did gdrdrv_open
    if (info->pid != task_pid(current)) {
        gdr_err("filp is not opened by the current process\n");
        return -EACCES;
    }

    handle = gdrdrv_handle_from_off(vma->vm_pgoff);
    mr = gdr_mr_from_handle(info, handle);
    // BUG: mr needs locking
    if (!mr) {
        gdr_dbg("cannot find handle in mr_list\n");
        ret = -EINVAL;
        goto out;
    }
    offset = mr->offset;
    if (gdr_mr_is_mapped(mr)) {
        gdr_dbg("mr has been mapped already\n");
        ret = -EINVAL;
        goto out;
    }
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
    if (offset > GPU_PAGE_SIZE * mr->page_table->entries) {
        gdr_dbg("offset %llu too big\n", offset);
        ret = -EINVAL;
        goto out;
    }
    if (size + offset > GPU_PAGE_SIZE * mr->page_table->entries) {
        gdr_dbg("size %zu too big\n", size);
        ret = -EINVAL;
        goto out;
    }
    if (size % PAGE_SIZE != 0) {
        gdr_dbg("size is not multiple of PAGE_SIZE\n");
    }
    // let's assume this mapping is not WC
    // this also works as the mapped flag for this mr
    mr->cpu_mapping_type = GDR_MR_CACHING;
    vma->vm_ops = &gdrdrv_vm_ops;
    gdr_dbg("overwriting vma->vm_private_data=0x%px with mr=0x%px\n", vma->vm_private_data, mr);
    vma->vm_private_data = mr;
    p = 0;
    vaddr = vma->vm_start;
    do {
        // map individual physically contiguous IO ranges
        unsigned long paddr = mr->page_table->pages[p]->physical_address;
        unsigned nentries = 1;
        size_t len;
        int is_wcomb;

        gdr_dbg("range start with p=%d vaddr=%lx page_paddr=%lx\n", p, vaddr, paddr);

        ++p;
        // check p-1 and p for contiguity
        {
            unsigned long prev_page_paddr = mr->page_table->pages[p-1]->physical_address;
            for(; p < mr->page_table->entries; ++p) {
                struct nvidia_p2p_page *page = mr->page_table->pages[p];
                unsigned long cur_page_paddr = page->physical_address;
                //gdr_dbg("p=%d prev_page_paddr=%lx cur_page_paddr=%lx\n",
                //        p, prev_page_paddr, cur_page_paddr);
                if (prev_page_paddr + GPU_PAGE_SIZE != cur_page_paddr) {
                    gdr_dbg("non-contig p=%d prev_page_paddr=%lx cur_page_paddr=%lx\n",
                            p, prev_page_paddr, cur_page_paddr);
                    break;
                }
                prev_page_paddr = cur_page_paddr;
                ++nentries;
            }
        }
        // offset not supported, see check above
        len = MIN(size, GPU_PAGE_SIZE * nentries);
        // phys range is [paddr, paddr+len-1]
        gdr_dbg("mapping p=%u entries=%d offset=%llx len=%zu vaddr=%lx paddr=%lx\n", 
                p, nentries, offset, len, vaddr, paddr);
        if (gdr_pfn_is_ram(paddr >> PAGE_SHIFT)) {
            WARN_ON_ONCE(!gdrdrv_cpu_can_cache_gpu_mappings);
            is_wcomb = 0;
        } else {
            is_wcomb = 1;
            // flagging the whole mr as a WC mapping if at least one chunk is WC
            mr->cpu_mapping_type = GDR_MR_WC;
        }
        ret = gdrdrv_remap_gpu_mem(vma, vaddr, paddr, len, is_wcomb);
        if (ret) {
            gdr_err("error %d in gdrdrv_remap_gpu_mem\n", ret);
            goto out;
        }
        vaddr += len;
        size -= len;
        offset = 0;
    } while(size && p < mr->page_table->entries);

    if (vaddr != vma->vm_end) {
        gdr_err("vaddr=%lx != vm_end=%lx\n", vaddr, vma->vm_end);
        ret = -EINVAL;
    }

out:
    if (ret) {
        if (mr) {
            mr->vma = NULL;
            mr->mapping = NULL;
            mr->cpu_mapping_type = GDR_MR_NONE;
            // TODO: tear down stale partial mappings
        }
    } else {
        mr->vma = vma;
        mr->mapping = filp->f_mapping;
        gdr_dbg("mr vma=0x%px mapping=0x%px\n", mr->vma, mr->mapping);
    }
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

#if defined(CONFIG_PPC64) && defined(PVR_POWER9)
    if (pvr_version_is(PVR_POWER9)) {
        // Approximating CPU-GPU coherence with CPU model
        // This might break in the future
        // A better way would be to detect the presence of the IBM-NPU bridges and
        // verify that all GPUs are connected through those
        gdrdrv_cpu_can_cache_gpu_mappings = 1;
    }
#endif

    if (gdrdrv_cpu_can_cache_gpu_mappings)
        gdr_msg(KERN_INFO, "enabling use of CPU cached mappings\n");

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
