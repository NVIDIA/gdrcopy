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
#include <linux/sched.h>
#include <linux/timex.h>
#include <linux/timer.h>
#include <linux/pci.h>
#include <linux/proc_fs.h>
#include <linux/seq_file.h>

#if LINUX_VERSION_CODE >= KERNEL_VERSION(4,11,0)
#include <linux/sched/signal.h>
#endif

/**
 * This is for PAT-awareness. pat_enabled is a GPL-only symbol. We can link only
 * if all relevant licenses permit.
 */
#if LINUX_VERSION_CODE >= KERNEL_VERSION(5,13,0) && defined(CONFIG_X86_64)
#define GDRDRV_KERNEL_MAY_USE_PAT 1

#ifdef GDRDRV_OPENSOURCE_NVIDIA
#define GDRDRV_CAN_CALL_PAT_ENABLED 1
extern bool pat_enabled(void);
#else
#define GDRDRV_CAN_CALL_PAT_ENABLED 0
#endif

#else
#define GDRDRV_KERNEL_MAY_USE_PAT 0
#define GDRDRV_CAN_CALL_PAT_ENABLED 0
#endif


/**
 * This is needed for round_up()
 */
#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 11, 0)
#include <linux/math.h>
#endif

/**
 * HAVE_UNLOCKED_IOCTL has been dropped in kernel version 5.9.
 * There is a chance that the removal might be ported back to 5.x.
 * So if HAVE_UNLOCKED_IOCTL is not defined in kernel v5, we define it.
 * This also allows backward-compatibility with kernel < 2.6.11.
 */
#if LINUX_VERSION_CODE >= KERNEL_VERSION(5, 0, 0) && !defined(HAVE_UNLOCKED_IOCTL)
#define HAVE_UNLOCKED_IOCTL 1
#endif 

/**
 * Old Linux kernel does not define this macro.
 */
#ifndef fallthrough
# define fallthrough    do {} while (0)  /* fallthrough */
#endif

//-----------------------------------------------------------------------------

static const unsigned int GDRDRV_BF3_PCI_ROOT_DEV_VENDOR_ID = 0x15b3;
static const unsigned int GDRDRV_BF3_PCI_ROOT_DEV_DEVICE_ID[2] = {0xa2da, 0xa2db};

//-----------------------------------------------------------------------------

static int gdrdrv_major = 0;
static int gdrdrv_cpu_can_cache_gpu_mappings = 0;
static int gdrdrv_cpu_must_use_device_mapping = 0;

//-----------------------------------------------------------------------------

static atomic64_t gdrdrv_nv_get_pages_refcount = ATOMIC_INIT(0);

//-----------------------------------------------------------------------------

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

static inline bool gdrdrv_pat_enabled(void)
{
#if GDRDRV_CAN_CALL_PAT_ENABLED
    // pat_enabled is GPL.
    return pat_enabled();
#elif GDRDRV_KERNEL_MAY_USE_PAT
    // We cannot use pat_enabled here because it is GPL.
    // Assume that it is enabled to handle the worst case scenario.
    return true;
#else
    // PAT is used on x86 only. It has been introduced in Linux v4.2. However, we
    // check for PAT from Linux v5.13. See gdrdrv_io_remap_pfn_range for more info.
    return false;
#endif
}

#ifndef GDRDRV_HAVE_VM_FLAGS_SET
/**
 * This API requires Linux kernel 6.3.
 * See https://github.com/torvalds/linux/commit/bc292ab00f6c7a661a8a605c714e8a148f629ef6
 */
static inline void vm_flags_set(struct vm_area_struct *vma, vm_flags_t flags)
{
    vma->vm_flags |= flags;
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
static inline pgprot_t pgprot_modify_device(pgprot_t old_prot)
{
    // Device mapping should never be called on x86
    BUG_ON(1);
    return old_prot;
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
static inline pgprot_t pgprot_modify_device(pgprot_t old_prot)
{
    // Device mapping should never be called on PPC64
    BUG_ON(1);
    return old_prot;
}
#define get_tsc_khz() (get_cycles()/1000) // dirty hack
static inline int gdr_pfn_is_ram(unsigned long pfn)
{
    // catch platforms, e.g. POWER8, POWER9 with GPUs not attached via NVLink,
    // where GPU memory is non-coherent
#ifdef GDRDRV_OPENSOURCE_NVIDIA
    // page_is_ram is a GPL symbol. We can use it with the open flavor of NVIDIA driver.
    return page_is_ram(pfn);
#else
    // For the proprietary flavor, we approximate using the following algorithm.
    unsigned long start = pfn << PAGE_SHIFT;
    unsigned long mask_47bits = (1UL<<47)-1;
    return gdrdrv_cpu_can_cache_gpu_mappings && (0 == (start & ~mask_47bits));
#endif
}

#elif defined(CONFIG_ARM64)
static inline pgprot_t pgprot_modify_writecombine(pgprot_t old_prot)
{
    return pgprot_writecombine(old_prot);
}
static inline pgprot_t pgprot_modify_device(pgprot_t old_prot)
{
    return pgprot_device(old_prot);
}
static inline int gdr_pfn_is_ram(unsigned long pfn)
{
#ifdef GDRDRV_OPENSOURCE_NVIDIA
    // page_is_ram is a GPL symbol. We can use it with the open flavor.
    return page_is_ram(pfn);
#else
    // For the proprietary flavor of NVIDIA driver, we use WC mapping.
    return 0;
#endif
}

#else
#error "X86_64/32 or PPC64 or ARM64 is required"
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

#ifdef GDRDRV_OPENSOURCE_NVIDIA
#define GDRDRV_BUILT_FOR_NVIDIA_FLAVOR_STRING "opensource"
#else
#define GDRDRV_BUILT_FOR_NVIDIA_FLAVOR_STRING "proprietary"
#endif

//-----------------------------------------------------------------------------

#define DEVNAME "gdrdrv"

#define gdr_msg(KRNLVL, FMT, ARGS...) printk(KRNLVL DEVNAME ":%s:" FMT, __func__, ## ARGS)
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

static int use_persistent_mapping = 1;

static struct proc_dir_entry *gdrdrv_proc_dir_entry = NULL;

//-----------------------------------------------------------------------------

MODULE_AUTHOR("drossetti@nvidia.com");
MODULE_LICENSE("Dual MIT/GPL");
MODULE_DESCRIPTION("GDRCopy kernel-mode driver built for " GDRDRV_BUILT_FOR_NVIDIA_FLAVOR_STRING " NVIDIA driver");
MODULE_VERSION(GDRDRV_VERSION_STRING);
module_param(dbg_enabled, int, 0000);
MODULE_PARM_DESC(dbg_enabled, "enable debug tracing");
module_param(info_enabled, int, 0000);
MODULE_PARM_DESC(info_enabled, "enable info tracing");
module_param(use_persistent_mapping, int, 0000);
MODULE_PARM_DESC(use_persistent_mapping, "use persistent mapping instead of traditional (non-persistent) mapping");

//-----------------------------------------------------------------------------

#define GPU_PAGE_SHIFT   16
#define GPU_PAGE_SIZE    ((u64)1 << GPU_PAGE_SHIFT)
#define GPU_PAGE_OFFSET  (GPU_PAGE_SIZE-1)
#define GPU_PAGE_MASK    (~GPU_PAGE_OFFSET)

#ifndef MAX
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#endif

#ifndef MIN
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#endif


// compatibility with old Linux kernels

#ifndef ACCESS_ONCE
#define ACCESS_ONCE(x) (*(volatile typeof(x) *)&(x))
#endif

#ifndef READ_ONCE
#define READ_ONCE(x) ACCESS_ONCE(x)
#endif

//-----------------------------------------------------------------------------

struct gdr_mr {
    struct list_head node;
    gdr_hnd_t handle;
    u64 offset;
    u64 p2p_token;
    u32 va_space;
    u32 page_size;
    u64 va;
    u64 mapped_size;
    gdr_mr_type_t cpu_mapping_type;
    gdr_mr_type_t req_mapping_type;
    nvidia_p2p_page_table_t *page_table;
    int cb_flag;
    cycles_t tm_cycles;
    unsigned int tsc_khz;
    struct vm_area_struct *vma;
    struct address_space *mapping;
    struct rw_semaphore sem;
    unsigned force_pci:1;
};
typedef struct gdr_mr gdr_mr_t;

/**
 * Prerequisite:
 * - mr must be protected by down_read(mr->sem) or stronger.
 */
static int gdr_mr_is_mapped(gdr_mr_t *mr)
{
    return mr->cpu_mapping_type != GDR_MR_NONE;
}

static inline void gdrdrv_zap_vma(struct address_space *mapping, struct vm_area_struct *vma)
{
    // This function is mainly used for files and the address is relative to
    // the file offset. We use vma->pg_off here to unmap this entire range but
    // not the other mapped ranges.
    unmap_mapping_range(mapping, vma->vm_pgoff << PAGE_SHIFT, vma->vm_end - vma->vm_start, 0);
}

/**
 * Prerequisite:
 * - mr must be protected by down_write(mr->sem).
 */
static void gdr_mr_destroy_all_mappings(gdr_mr_t *mr)
{
    // there is a single mapping at the moment
    if (mr->vma)
        gdrdrv_zap_vma(mr->mapping, mr->vma);

    mr->cpu_mapping_type = GDR_MR_NONE;
}

//-----------------------------------------------------------------------------

struct gdr_info {
    // simple low-performance linked-list implementation
    struct list_head        mr_list;
    struct mutex            lock;

    // Pointer to the pid struct of the creator task group.
    // We do not use numerical pid here to avoid issues from pid reuse.
    struct pid             *tgid;

    // Address space unique to this opened file. We need to create a new one
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

static int gdrdrv_check_same_process(gdr_info_t *info, struct task_struct *tsk)
{
    int same_proc;
    BUG_ON(0 == info);
    BUG_ON(0 == tsk);
    same_proc = (info->tgid == task_tgid(tsk)) ; // these tasks belong to the same task group
    if (!same_proc) {
        gdr_dbg("check failed, info:{tgid=%p} this tsk={tgid=%p}\n",
                info->tgid, task_tgid(tsk));
    }
    return same_proc;
}

//-----------------------------------------------------------------------------

static inline int gdr_support_persistent_mapping(void)
{
#if defined(NVIDIA_P2P_CAP_GET_PAGES_PERSISTENT_API)
    return 1;
#elif defined(NVIDIA_P2P_CAP_PERSISTENT_PAGES)
    return !!(nvidia_p2p_cap_persistent_pages);
#else
    return 0;
#endif
}

static inline int gdr_use_persistent_mapping(void)
{
    return use_persistent_mapping && gdr_support_persistent_mapping();
}

//-----------------------------------------------------------------------------

static inline int gdr_support_force_pcie(void)
{
#ifdef NVIDIA_P2P_FLAGS_FORCE_BAR1_MAPPING
    return gdr_use_persistent_mapping();
#else
    return 0;
#endif
}

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
    // here we track the task group owning the driver fd and prevent other processes
    // to use it.
    info->tgid = task_tgid(current);

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

/**
 * Clean up and free all resources (e.g., page_table) associated with this mr.
 *
 * Prerequisites:
 * - mr->sem must be under down_write before calling this function.
 * - There is no mapping associated with this mr.
 *
 * After this function returns, mr is freed and cannot be accessed anymore.
 *
 */
static void gdr_free_mr_unlocked(gdr_mr_t *mr)
{
    int status = 0;
    nvidia_p2p_page_table_t *page_table = NULL;

    BUG_ON(!mr);
    BUG_ON(gdr_mr_is_mapped(mr));

    page_table = mr->page_table;
    if (page_table) {
        gdr_info("invoking nvidia_p2p_put_pages(va=0x%llx p2p_tok=%llx va_tok=%x)\n",
                 mr->va, mr->p2p_token, mr->va_space);

        // We reach here before gdrdrv_get_pages_free_callback.
        // However, it might be waiting on semaphore.
        // Release the semaphore to let it progresses.
        up_write(&mr->sem);

        // In case gdrdrv_get_pages_free_callback is inflight, nvidia_p2p_put_pages will be blocked.
        #ifdef NVIDIA_P2P_CAP_GET_PAGES_PERSISTENT_API
        if (gdr_use_persistent_mapping()) {
            status = nvidia_p2p_put_pages_persistent(mr->va, page_table, 0);
            if (status) {
                gdr_err("nvidia_p2p_put_pages_persistent error %d\n", status);
            }
        } else {
            status = nvidia_p2p_put_pages(mr->p2p_token, mr->va_space, mr->va, page_table);
            if (status) {
                gdr_err("nvidia_p2p_put_pages error %d, async callback may have been fired\n", status);
            }
        }
        #else
        status = nvidia_p2p_put_pages(mr->p2p_token, mr->va_space, mr->va, page_table);
        if (status) {
            gdr_err("nvidia_p2p_put_pages error %d, async callback may have been fired\n", status);
        }
        #endif

        if (!status) {
            atomic64_fetch_dec(&gdrdrv_nv_get_pages_refcount);
        }
    } else {
        gdr_dbg("invoking unpin_buffer while callback has already been fired\n");

        // From this point, no other code paths will access this mr.
        // We release semaphore and clear the mr.
        up_write(&mr->sem);
    }

    memset(mr, 0, sizeof(*mr));
    kfree(mr);
}


//-----------------------------------------------------------------------------

static int gdrdrv_release(struct inode *inode, struct file *filp)
{
    gdr_info_t *info = filp->private_data;
    gdr_mr_t *mr = NULL;
    nvidia_p2p_page_table_t *page_table = NULL;
    struct list_head *p, *n;

    gdr_dbg("closing\n");

    if (!info) {
        gdr_err("filp contains no info\n");
        return -EIO;
    }

    mutex_lock(&info->lock);
    list_for_each_safe(p, n, &info->mr_list) {
        page_table = NULL;

        mr = list_entry(p, gdr_mr_t, node);

        down_write(&mr->sem);
        gdr_info("freeing MR=0x%px\n", mr);

        if (gdr_mr_is_mapped(mr)) {
            gdr_mr_destroy_all_mappings(mr);
        }

        list_del(&mr->node);

        gdr_free_mr_unlocked(mr);
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

/** 
 * Convert handle to mr and semaphore-acquire it with read or write.
 * If success, that mr is guaranteed to be available until gdr_put_mr is called.
 * On success, return mr. Otherwise, return NULL.
 */
static inline gdr_mr_t *gdr_get_mr_from_handle(gdr_info_t *info, gdr_hnd_t handle, int write)
{
    gdr_mr_t *mr;
    mutex_lock(&info->lock);
    mr = gdr_mr_from_handle_unlocked(info, handle);
    if (mr) {
        if (write)
            down_write(&mr->sem);
        else
            down_read(&mr->sem);
    }
    mutex_unlock(&info->lock);
    return mr;
}

#define gdr_get_mr_from_handle_read(info, handle)   (gdr_get_mr_from_handle((info), (handle), 0))
#define gdr_get_mr_from_handle_write(info, handle)  (gdr_get_mr_from_handle((info), (handle), 1))

//-----------------------------------------------------------------------------

/**
 * Put the mr object. The `write` parameter must match the previous gdr_get_mr_from_handle call.
 * After this function returns, mr may cease to exist (freed). It must not be accessed again.
 */
static inline void gdr_put_mr(gdr_mr_t *mr, int write)
{
    if (write)
        up_write(&mr->sem);
    else
        up_read(&mr->sem);
}

#define gdr_put_mr_read(mr)     (gdr_put_mr((mr), 0))
#define gdr_put_mr_write(mr)    (gdr_put_mr((mr), 1))

//-----------------------------------------------------------------------------
// off is host page aligned, because of the kernel interface
// could abuse extra available bits for other purposes

static gdr_hnd_t gdrdrv_handle_from_off(unsigned long off)
{
    return (gdr_hnd_t)(off);
}

//-----------------------------------------------------------------------------

typedef void (*gdr_free_callback_fn_t)(void *);

static void gdrdrv_get_pages_free_callback(void *data)
{
    gdr_mr_t *mr = data;
    nvidia_p2p_page_table_t *page_table = NULL;
    gdr_info("free callback\n");
    // can't take the info->lock here due to potential AB-BA
    // deadlock with internal NV driver lock(s)
    down_write(&mr->sem);
    mr->cb_flag = 1;
    page_table = mr->page_table;
    if (page_table) {
        nvidia_p2p_free_page_table(page_table);
        // NVIDIA driver takes back those pages. Decrement the refcount here.
        atomic64_fetch_dec(&gdrdrv_nv_get_pages_refcount);
        if (gdr_mr_is_mapped(mr))
            gdr_mr_destroy_all_mappings(mr);
    } else {
        gdr_dbg("free callback, page_table is NULL\n");
    }
    mr->page_table = NULL;
    up_write(&mr->sem);
}

//-----------------------------------------------------------------------------

/**
 * Generate mr->handle. This function should be called under info->lock.
 *
 * Prerequisite:
 * - mr->mapped_size is set and round to max(PAGE_SIZE, GPU_PAGE_SIZE)
 * - mr->sem must be under down_write before calling this function.
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
    
    next_handle = info->next_handle + MAX(1, mr->mapped_size >> PAGE_SHIFT);

    // The next handle will be overflowed, so we mark it.
    if (unlikely((next_handle & ((gdr_hnd_t)(-1) >> PAGE_SHIFT)) < info->next_handle))
        info->next_handle_overflow = 1;

    mr->handle = info->next_handle;
    info->next_handle = next_handle;

    return 0;
}

//-----------------------------------------------------------------------------

static int __gdrdrv_pin_buffer(gdr_info_t *info, u64 addr, u64 size, u64 p2p_token, u32 va_space, u32 flags, gdr_hnd_t *p_handle)
{
    int ret = 0;
    struct nvidia_p2p_page_table *page_table = NULL;
    u64 page_virt_start;
    u64 page_virt_end;
    size_t rounded_size;
    gdr_mr_t *mr = NULL;
    gdr_free_callback_fn_t free_callback_fn;
    #ifndef CONFIG_ARM64
    cycles_t ta, tb;
    #endif
    u32 get_pages_flags;

    mr = kmalloc(sizeof(gdr_mr_t), GFP_KERNEL);
    if (!mr) {
        gdr_err("can't alloc kernel memory\n");
        ret = -ENOMEM;
        goto out;
    }
    memset(mr, 0, sizeof(*mr));

    // do proper alignment, as required by NVIDIA driver.
    // align both size and addr as it is a requirement of nvidia_p2p_get_pages* API
    page_virt_start  = addr & GPU_PAGE_MASK;
    page_virt_end    = round_up((addr + size), GPU_PAGE_SIZE);
    rounded_size     = page_virt_end - page_virt_start;

    init_rwsem(&mr->sem);

    free_callback_fn = gdr_use_persistent_mapping() ? NULL : gdrdrv_get_pages_free_callback;

    get_pages_flags = 0;
    mr->offset       = addr & GPU_PAGE_OFFSET;
    if (!gdr_use_persistent_mapping()) {
        if (GDRDRV_PIN_BUFFER_FLAG_DEFAULT != flags) {
            gdr_err("flags are not supported in non persistent mode\n");
            ret = -EINVAL;
            goto out;
        }
        mr->p2p_token    = p2p_token;
        mr->va_space     = va_space;
    } else {
        if (p2p_token || va_space) {
            gdr_err("p2p token and va space are deprecated and now unsupported\n");
            ret = -EINVAL;
            goto out;
        }
        mr->p2p_token    = 0;
        mr->va_space     = 0;
        if (flags & GDRDRV_PIN_BUFFER_FLAG_FORCE_PCIE) {
#ifdef NVIDIA_P2P_FLAGS_FORCE_BAR1_MAPPING
            get_pages_flags |= NVIDIA_P2P_FLAGS_FORCE_BAR1_MAPPING;
            mr->force_pci = 1;
#else
            gdr_err("The GPU driver used to build this module did not have the NVIDIA_P2P_FLAGS_FORCE_BAR1_MAPPING flag\n");
            ret = -EINVAL;
            goto out;
#endif
        }
    }
    mr->va           = page_virt_start;
    mr->mapped_size  = rounded_size;
    mr->cpu_mapping_type = GDR_MR_NONE;
    mr->page_table   = NULL;
    mr->cb_flag      = 0;

    #ifndef CONFIG_ARM64
    ta = get_cycles();
    #endif

    // After nvidia_p2p_get_pages returns (successfully), gdrdrv_get_pages_free_callback may be invoked anytime.
    // mr setup must be done before calling that API. The memory barrier is included in down_write.

    // We take this semaphore to prevent race with gdrdrv_get_pages_free_callback.
    down_write(&mr->sem);

    #ifdef NVIDIA_P2P_CAP_GET_PAGES_PERSISTENT_API
    if (free_callback_fn) {
        ret = nvidia_p2p_get_pages(mr->p2p_token, mr->va_space, mr->va, mr->mapped_size, &page_table,
                                   free_callback_fn, mr);
        gdr_info("invoking nvidia_p2p_get_pages(va=0x%llx len=%lld p2p_tok=%llx va_tok=%x callback=%px)\n",
                 mr->va, mr->mapped_size, mr->p2p_token, mr->va_space, free_callback_fn);
    } else {
        ret = nvidia_p2p_get_pages_persistent(mr->va, mr->mapped_size, &page_table, get_pages_flags);
        gdr_info("invoking nvidia_p2p_get_pages_persistent(va=0x%llx len=%lld)\n",
                 mr->va, mr->mapped_size);
    }
    #else
    ret = nvidia_p2p_get_pages(mr->p2p_token, mr->va_space, mr->va, mr->mapped_size, &page_table,
                               free_callback_fn, mr);
    gdr_info("invoking nvidia_p2p_get_pages(va=0x%llx len=%lld p2p_tok=%llx va_tok=%x callback=%px)\n",
             mr->va, mr->mapped_size, mr->p2p_token, mr->va_space, free_callback_fn);
    #endif

    if (!ret) {
        atomic64_fetch_inc(&gdrdrv_nv_get_pages_refcount);
    }

    #ifndef CONFIG_ARM64
    tb = get_cycles();
    #endif
    if (ret < 0) {
        gdr_err("nvidia_p2p_get_pages(va=%llx len=%lld p2p_token=%llx va_space=%x callback=%px) failed [ret = %d]\n",
                mr->va, mr->mapped_size, mr->p2p_token, mr->va_space, free_callback_fn, ret);
        goto out;
    }
    mr->page_table = page_table;
    #ifndef CONFIG_ARM64
    mr->tm_cycles = tb - ta;
    mr->tsc_khz = get_tsc_khz();
    #endif


    // check version before accessing page table
    if (!NVIDIA_P2P_PAGE_TABLE_VERSION_COMPATIBLE(page_table)) {
        gdr_err("incompatible page table version 0x%08x\n", page_table->version);
        ret = -EFAULT;
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

    if (!ret) {
        list_add(&mr->node, &info->mr_list);
        *p_handle = mr->handle;
        up_write(&mr->sem);
    }
    mutex_unlock(&info->lock);


out:
    if (ret && mr) {
        gdr_free_mr_unlocked(mr);
        mr = NULL;
    }

    return ret;
}

//-----------------------------------------------------------------------------

static int __gdrdrv_unpin_buffer(gdr_info_t *info, gdr_hnd_t handle)
{
    int ret = 0;

    gdr_mr_t *mr = NULL;

    // someone might try to traverse the list and/or to do something
    // to the mr at the same time, so let's lock here
    mutex_lock(&info->lock);
    mr = gdr_mr_from_handle_unlocked(info, handle);
    if (NULL == mr) {
        gdr_err("unexpected handle %llx while unmapping buffer\n", handle);
        ret = -EINVAL;
    } else {
        // Found the mr. Let's lock it.
        down_write(&mr->sem);
        if (gdr_mr_is_mapped(mr)) {
            gdr_mr_destroy_all_mappings(mr);
        }

        // Remove this handle from the list under info->lock.
        // Now race with gdrdrv_get_pages_free_callback is the only thing we need to care about.
        list_del(&mr->node);
    }
    mutex_unlock(&info->lock);

    if (ret)
        goto out;

    gdr_free_mr_unlocked(mr);

 out:
    return ret;
}

//-----------------------------------------------------------------------------

static int gdrdrv_pin_buffer(gdr_info_t *info, void __user *_params)
{
    int ret = 0;

    struct GDRDRV_IOC_PIN_BUFFER_PARAMS params = {0};

    int has_handle = 0;
    gdr_hnd_t handle;

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

    ret = __gdrdrv_pin_buffer(info, params.addr, params.size, params.p2p_token, params.va_space, GDRDRV_PIN_BUFFER_FLAG_DEFAULT, &handle);
    if (ret)
        goto out;

    has_handle = 1;
    params.handle = handle;

    if (copy_to_user(_params, &params, sizeof(params))) {
        gdr_err("copy_to_user failed on user pointer 0x%px\n", _params);
        ret = -EFAULT;
    }


out:
    if (ret) {
        if (has_handle)
            __gdrdrv_unpin_buffer(info, handle);
    }

    return ret;
}

//-----------------------------------------------------------------------------

static int gdrdrv_pin_buffer_v2(gdr_info_t *info, void __user *_params)
{
    int ret = 0;

    struct GDRDRV_IOC_PIN_BUFFER_V2_PARAMS params = {0};

    int has_handle = 0;
    gdr_hnd_t handle;

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

    if (params.flags & ~(GDRDRV_PIN_BUFFER_FLAG_DEFAULT|GDRDRV_PIN_BUFFER_FLAG_FORCE_PCIE)) {
        gdr_err("invalid flags\n");
        ret = -EINVAL;
        goto out;
    }

    ret = __gdrdrv_pin_buffer(info, params.addr, params.size, 0, 0, params.flags, &handle);
    if (ret)
        goto out;

    has_handle = 1;
    params.handle = handle;

    if (copy_to_user(_params, &params, sizeof(params))) {
        gdr_err("copy_to_user failed on user pointer 0x%px\n", _params);
        ret = -EFAULT;
    }


out:
    if (ret) {
        if (has_handle)
            __gdrdrv_unpin_buffer(info, handle);
    }

    return ret;
}

//-----------------------------------------------------------------------------

static int gdrdrv_unpin_buffer(gdr_info_t *info, void __user *_params)
{
    struct GDRDRV_IOC_UNPIN_BUFFER_PARAMS params = {0};
    int ret = 0;

    if (copy_from_user(&params, _params, sizeof(params))) {
        gdr_err("copy_from_user failed on user pointer 0x%px\n", _params);
        return -EFAULT;
    }

    ret = __gdrdrv_unpin_buffer(info, params.handle);

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

    mr = gdr_get_mr_from_handle_read(info, params.handle);
    if (NULL == mr) {
        gdr_err("unexpected handle %llx in get_cb_flag\n", params.handle);
        ret = -EINVAL;
        goto out;
    }

    params.flag = !!(mr->cb_flag);

    gdr_put_mr_read(mr);

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

    mr = gdr_get_mr_from_handle_read(info, params.handle);
    if (NULL == mr) {
        gdr_err("unexpected handle %llx in get_cb_flag\n", params.handle);
        ret = -EINVAL;
        goto out;
    }

    params.va           = mr->va;
    params.mapped_size  = mr->mapped_size;
    params.page_size    = mr->page_size;
    params.tm_cycles    = mr->tm_cycles;
    params.tsc_khz      = mr->tsc_khz;
    params.mapped       = gdr_mr_is_mapped(mr);
    params.wc_mapping   = (mr->cpu_mapping_type == GDR_MR_WC);
    params.paddr        = mr->page_table->pages[0]->physical_address;

    gdr_put_mr_read(mr);

    if (copy_to_user(_params, &params, sizeof(params))) {
        gdr_err("copy_to_user failed on user pointer 0x%px\n", _params);
        ret = -EFAULT;
    }
 out:
    return ret;
}

//-----------------------------------------------------------------------------

static int gdrdrv_get_info_v2(gdr_info_t *info, void __user *_params)
{
    struct GDRDRV_IOC_GET_INFO_V2_PARAMS params = {0};
    int ret = 0;
    gdr_mr_t *mr = NULL;

    if (copy_from_user(&params, _params, sizeof(params))) {
        gdr_err("copy_from_user failed on user pointer 0x%px\n", _params);
        ret = -EFAULT;
        goto out;
    }

    mr = gdr_get_mr_from_handle_read(info, params.handle);
    if (NULL == mr) {
        gdr_err("unexpected handle %llx in get_cb_flag\n", params.handle);
        ret = -EINVAL;
        goto out;
    }

    params.va           = mr->va;
    params.mapped_size  = mr->mapped_size;
    params.page_size    = mr->page_size;
    params.tm_cycles    = mr->tm_cycles;
    params.tsc_khz      = mr->tsc_khz;
    params.mapping_type = mr->cpu_mapping_type;
    params.paddr        = mr->page_table->pages[0]->physical_address;

    gdr_put_mr_read(mr);

    if (copy_to_user(_params, &params, sizeof(params))) {
        gdr_err("copy_to_user failed on user pointer 0x%px\n", _params);
        ret = -EFAULT;
    }
 out:
    return ret;
}

//-----------------------------------------------------------------------------

static int gdrdrv_get_attr(gdr_info_t *info, void __user *_params)
{
    struct GDRDRV_IOC_GET_ATTR_PARAMS params = {0};
    int ret = 0;

    if (copy_from_user(&params, _params, sizeof(params))) {
        gdr_err("copy_from_user failed on user pointer 0x%px\n", _params);
        ret = -EFAULT;
        goto out;
    }

    switch (params.attr) {
    case GDRDRV_ATTR_USE_PERSISTENT_MAPPING:
        params.val = gdr_use_persistent_mapping();
        break;
    case GDRDRV_ATTR_SUPPORT_PIN_FLAG_FORCE_PCIE:
        params.val = gdr_support_force_pcie();
        break;
    default:
        ret = -EINVAL;
    }

    if (copy_to_user(_params, &params, sizeof(params))) {
        gdr_err("copy_to_user failed on user pointer 0x%px\n", _params);
        ret = -EFAULT;
    }

 out:
    return ret;
}

//-----------------------------------------------------------------------------

static int gdrdrv_req_mapping_type(gdr_info_t *info, void __user *_params)
{
    struct GDRDRV_IOC_REQ_MAPPING_TYPE_PARAMS params = {0};
    int ret = 0;

    gdr_mr_t *mr = NULL;

    if (copy_from_user(&params, _params, sizeof(params))) {
        gdr_err("copy_from_user failed on user pointer 0x%px\n", _params);
        ret = -EFAULT;
        goto out;
    }

    if (GDR_MR_NONE > params.mapping_type || GDR_MR_DEVICE < params.mapping_type) {
        gdr_err("Unknown request_mapping_type=%d\n", params.mapping_type);
        ret = -EINVAL;
        goto out;
    }

    mr = gdr_get_mr_from_handle_write(info, params.handle);
    if (NULL == mr) {
        gdr_err("Unexpected handle %llx in req_mapping_type\n", params.handle);
        ret = -EINVAL;
        goto out;
    }

    // Changing the mapping type after the mr is mapped is not allowed.
    if (gdr_mr_is_mapped(mr)) {
        ret = -EBUSY;
        goto out;
    }

    switch (params.mapping_type) {
        case GDR_MR_CACHING:
            // All pages are either RAM or BAR1. We don't need to check every page.
            // Just in case there is a bug that causes entries == 0, we also check that before accessing pages[0].
            // In this case, we won't do mapping. So, it does not matter if we set req_mapping_type to GDR_MR_CACHING.
            if (!gdrdrv_cpu_can_cache_gpu_mappings
                || (mr->page_table->entries > 0 && !gdr_pfn_is_ram(mr->page_table->pages[0]->physical_address >> PAGE_SHIFT))
            ) {
                ret = -EOPNOTSUPP;
                goto out;
            }
            break;
        case GDR_MR_WC:
            if (gdrdrv_cpu_must_use_device_mapping) {
                ret = -EOPNOTSUPP;
                goto out;
            }
            break;
        case GDR_MR_DEVICE:
            if (!gdrdrv_cpu_must_use_device_mapping) {
                ret = -EOPNOTSUPP;
                goto out;
            }
            break;
        case GDR_MR_NONE:
            break;
        default:
            gdr_err("Unknown request_mapping_type=%d\n", params.mapping_type);
            ret = -EINVAL;
            goto out;
    }
    mr->req_mapping_type = params.mapping_type;

out:
    if (mr)
        gdr_put_mr_write(mr);
    return ret;
}

//-----------------------------------------------------------------------------

static int gdrdrv_get_version(gdr_info_t *info, void __user *_params)
{
    struct GDRDRV_IOC_GET_VERSION_PARAMS params = {0};
    int ret = 0;

    params.gdrdrv_version = GDRDRV_VERSION;
    params.minimum_gdr_api_version = MINIMUM_GDR_API_VERSION;

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

    if (!info) {
        gdr_err("filp contains no info\n");
        return -EIO;
    }
    // Check that the caller is the same process that did gdrdrv_open
    if (!gdrdrv_check_same_process(info, current)) {
        gdr_dbg("filp is not opened by the current process\n");
        return -EACCES;
    }

    switch (cmd) {
    case GDRDRV_IOC_PIN_BUFFER:
        ret = gdrdrv_pin_buffer(info, argp);
        break;

    case GDRDRV_IOC_PIN_BUFFER_V2:
        ret = gdrdrv_pin_buffer_v2(info, argp);
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

    case GDRDRV_IOC_GET_INFO_V2:
        ret = gdrdrv_get_info_v2(info, argp);
        break;

    case GDRDRV_IOC_GET_ATTR:
        ret = gdrdrv_get_attr(info, argp);
	break;

    case GDRDRV_IOC_REQ_MAPPING_TYPE:
        ret = gdrdrv_req_mapping_type(info, argp);
        break;

    case GDRDRV_IOC_GET_VERSION:
        ret = gdrdrv_get_version(info, argp);
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

static void gdrdrv_vma_close(struct vm_area_struct *vma)
{
    gdr_hnd_t handle;
    gdr_mr_t *mr = NULL;
    gdr_info_t* info = NULL;

    if (!vma->vm_file)
        return;

    info = vma->vm_file->private_data;
    if (!info)
        return;

    handle = gdrdrv_handle_from_off(vma->vm_pgoff);
    mr = gdr_get_mr_from_handle_write(info, handle);
    if (!mr)
        return;

    gdr_dbg("closing vma=0x%px vm_file=0x%px mr=0x%px mr->vma=0x%px\n", vma, vma->vm_file, mr, mr->vma);
    // TODO: handle multiple vma's
    mr->vma = NULL;
    mr->cpu_mapping_type = GDR_MR_NONE;
    gdr_put_mr_write(mr);
}

/*----------------------------------------------------------------------------*/

static const struct vm_operations_struct gdrdrv_vm_ops = {
    .close = gdrdrv_vma_close,
};

/*----------------------------------------------------------------------------*/

/**
 * Starting from kernel version 5.18-rc1, io_remap_pfn_range may use a GPL
 * function. This happens on x86 platforms that have
 * CONFIG_ARCH_HAS_CC_PLATFORM defined. The root cause is from pgprot_decrypted
 * implementation that has been changed to use cc_mkdec. To avoid the
 * GPL-incompatibility issue with the proprietary flavor of NVIDIA driver, we
 * reimplement io_remap_pfn_range according to the Linux kernel 5.17.15, which
 * predates support for Intel CC.
 */
static inline int __gdrdrv_io_remap_pfn_range(struct vm_area_struct *vma, unsigned long vaddr, unsigned long pfn, size_t size, pgprot_t prot)
{
#if defined(GDRDRV_OPENSOURCE_NVIDIA) || !((defined(CONFIG_X86_64) || defined(CONFIG_X86_32)) && IS_ENABLED(CONFIG_ARCH_HAS_CC_PLATFORM))
    return io_remap_pfn_range(vma, vaddr, pfn, size, prot);
#else

#ifndef CONFIG_AMD_MEM_ENCRYPT
#warning "CC is not fully functional in gdrdrv with the proprietary flavor of NVIDIA driver on Intel CPU. Use the open-source flavor if you want full support."
#endif

    return remap_pfn_range(vma, vaddr, pfn, size, __pgprot(__sme_clr(pgprot_val(prot))));
#endif
}

/*----------------------------------------------------------------------------*/

static inline int gdrdrv_io_remap_pfn_range(struct vm_area_struct *vma, unsigned long vaddr, unsigned long pfn, size_t size, pgprot_t prot)
{
    int status = 0;
#if GDRDRV_KERNEL_MAY_USE_PAT
    // Upstream Linux kernel has introduced track_pfn_remap in remap_pfn_range
    // since v5.13. On x86-64, PAT may be enabled. In that situation,
    // track_pfn_remap may fail if we do not call io_remap_pfn_range that
    // covers the entire VMA -- see
    // https://elixir.bootlin.com/linux/v5.13/source/arch/x86/mm/pat/memtype.c#L1047-L1055.
    // As a WAR, we call io_remap_pfn_range for each host `PAGE_SIZE` one by
    // one until we cover the entire `size`.
    BUG_ON(!vma);
    if (gdrdrv_pat_enabled() && !(vaddr == vma->vm_start && size == (vma->vm_end - vma->vm_start))) {
        size_t remaining_size = size;
        unsigned long chunk_pfn = pfn;
        unsigned long chunk_vaddr = vaddr;
        const unsigned long chunk_size = PAGE_SIZE;
        BUG_ON(size % PAGE_SIZE != 0);
        while (remaining_size > 0) {
            status = __gdrdrv_io_remap_pfn_range(vma, chunk_vaddr, chunk_pfn, chunk_size, prot);
            if (status) {
                gdr_err("Error %d in __gdrdrv_io_remap_pfn_range chunk_vaddr=%#lx, chunk_pfn=%#lx, chunk_size=%zu\n", status, chunk_vaddr, chunk_pfn, chunk_size);
                return status;
            }
            chunk_vaddr += chunk_size;
            ++chunk_pfn;
            remaining_size -= chunk_size;
        }
    }
    else
#endif
    {
        status = __gdrdrv_io_remap_pfn_range(vma, vaddr, pfn, size, prot);
    }

    return status;
}

/*----------------------------------------------------------------------------*/

static int gdrdrv_remap_gpu_mem(struct vm_area_struct *vma, unsigned long vaddr, unsigned long paddr, size_t size, gdr_mr_type_t mapping_type)
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

    // Disallow mmapped VMA to propagate to children processes
    vm_flags_set(vma, VM_DONTCOPY);

    if (mapping_type == GDR_MR_WC) {
        // override prot to create non-coherent WC mappings
        vma->vm_page_prot = pgprot_modify_writecombine(vma->vm_page_prot);
    } else if (mapping_type == GDR_MR_DEVICE) {
        // override prot to create non-coherent device mappings
        vma->vm_page_prot = pgprot_modify_device(vma->vm_page_prot);
    } else {
        // by default, vm_page_prot should be set to create cached mappings
    }
    if (gdrdrv_io_remap_pfn_range(vma, vaddr, pfn, size, vma->vm_page_prot)) {
        gdr_err("error in gdrdrv_io_remap_pfn_range()\n");
        ret = -EAGAIN;
        goto out;
    }

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
    gdr_mr_type_t cpu_mapping_type = GDR_MR_NONE;

    gdr_info("mmap filp=0x%px vma=0x%px vm_file=0x%px start=0x%lx size=%zu off=0x%lx\n", filp, vma, vma->vm_file, vma->vm_start, size, vma->vm_pgoff);

    if (!info) {
        gdr_err("filp contains no info\n");
        return -EIO;
    }
    // Check that the caller is the same process that did gdrdrv_open
    if (!gdrdrv_check_same_process(info, current)) {
        gdr_dbg("filp is not opened by the current process\n");
        return -EACCES;
    }

    handle = gdrdrv_handle_from_off(vma->vm_pgoff);
    mr = gdr_get_mr_from_handle_write(info, handle);
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

    // Set to None first
    mr->cpu_mapping_type = GDR_MR_NONE;
    vma->vm_ops = &gdrdrv_vm_ops;

    // check for physically contiguous IO ranges
    p = 0;
    vaddr = vma->vm_start;
    do {
        // map individual physically contiguous IO ranges
        unsigned long paddr = mr->page_table->pages[p]->physical_address;
        unsigned nentries = 1;
        size_t len;
        gdr_mr_type_t chunk_mapping_type = GDR_MR_NONE;

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
        if (mr->req_mapping_type == GDR_MR_NONE) {
            if (gdr_pfn_is_ram(paddr >> PAGE_SHIFT)) {
                WARN_ON_ONCE(!gdrdrv_cpu_can_cache_gpu_mappings);
                chunk_mapping_type = GDR_MR_CACHING;
            } else if (gdrdrv_cpu_must_use_device_mapping) {
                chunk_mapping_type = GDR_MR_DEVICE;
            } else {
                // flagging the whole mr as a WC mapping if at least one chunk is WC
                chunk_mapping_type = GDR_MR_WC;
            }
        } else
            chunk_mapping_type = mr->req_mapping_type;

        if (cpu_mapping_type == GDR_MR_NONE)
            cpu_mapping_type = chunk_mapping_type;

        // We don't handle when different chunks have different mapping types.
        // This scenario should never happen.
        BUG_ON(cpu_mapping_type != chunk_mapping_type);

        ret = gdrdrv_remap_gpu_mem(vma, vaddr, paddr, len, cpu_mapping_type);
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
        }
    } else {
        mr->vma = vma;
        mr->mapping = filp->f_mapping;

        BUG_ON(cpu_mapping_type == GDR_MR_NONE);
        // An mr with force_pci=1 should have not been mapped as cached on the CPU
        BUG_ON(mr->force_pci && (cpu_mapping_type == GDR_MR_CACHING));
        mr->cpu_mapping_type = cpu_mapping_type;

        gdr_dbg("mr vma=0x%px mapping=0x%px\n", mr->vma, mr->mapping);
    }

    if (mr)
        gdr_put_mr_write(mr);

    return ret;
}

//-----------------------------------------------------------------------------

static int gdrdrv_proc_params_read(struct seq_file *s, void *v)
{
    seq_printf(s, "Version: %s\n", GDRDRV_VERSION_STRING);
    seq_printf(s, "Built for NVIDIA driver flavor: %s\n", GDRDRV_BUILT_FOR_NVIDIA_FLAVOR_STRING);
    seq_printf(s, "Use persistent mapping: %s\n", gdr_use_persistent_mapping() ? "yes" : "no");
    seq_printf(s, "dbg_enabled: %d\n", dbg_enabled);
    seq_printf(s, "info_enabled: %d\n", info_enabled);
    return 0;
}

static int gdrdrv_proc_params_open(struct inode *inode, struct file *filp)
{
    return single_open(filp, gdrdrv_proc_params_read, NULL);
}

static int gdrdrv_proc_params_release(struct inode *inode, struct file *filp)
{
    return single_release(inode, filp);
}

#ifdef GDRDRV_HAVE_PROC_OPS
static const struct proc_ops gdrdrv_proc_params_proc_ops = {
    .proc_open = gdrdrv_proc_params_open,
    .proc_read = seq_read,
    .proc_lseek = seq_lseek,
    .proc_release = gdrdrv_proc_params_release
};
#else
static const struct file_operations gdrdrv_proc_params_proc_ops = {
    .open = gdrdrv_proc_params_open,
    .read = seq_read,
    .llseek = seq_lseek,
    .release = gdrdrv_proc_params_release
};
#endif

static int gdrdrv_proc_ngp_refcount_read(struct seq_file *s, void *v)
{
    seq_printf(s, "%lld\n", atomic64_read(&gdrdrv_nv_get_pages_refcount));
    return 0;
}

static int gdrdrv_proc_ngp_refcount_open(struct inode *inode, struct file *filp)
{
    return single_open(filp, gdrdrv_proc_ngp_refcount_read, NULL);
}

static int gdrdrv_proc_ngp_refcount_release(struct inode *inode, struct file *filp)
{
    return single_release(inode, filp);
}

#ifdef GDRDRV_HAVE_PROC_OPS
static const struct proc_ops gdrdrv_proc_ngp_refcount_proc_ops = {
    .proc_open = gdrdrv_proc_ngp_refcount_open,
    .proc_read = seq_read,
    .proc_lseek = seq_lseek,
    .proc_release = gdrdrv_proc_ngp_refcount_release
};
#else
static const struct file_operations gdrdrv_proc_ngp_refcount_proc_ops = {
    .open = gdrdrv_proc_ngp_refcount_open,
    .read = seq_read,
    .llseek = seq_lseek,
    .release = gdrdrv_proc_ngp_refcount_release
};
#endif

static int gdrdrv_procfs_init(void)
{
    int status = 0;
    char gdrdrv_proc_dir_name[20];
    int gdrdrv_proc_dir_mode = (S_IFDIR | S_IRUGO | S_IXUGO);

    struct proc_dir_entry *entry;
    int entry_mode = (S_IFREG | S_IRUGO);

    snprintf(gdrdrv_proc_dir_name, sizeof(gdrdrv_proc_dir_name), "driver/%s", DEVNAME);
    gdrdrv_proc_dir_name[sizeof(gdrdrv_proc_dir_name) - 1] = '\0';

    gdrdrv_proc_dir_entry = proc_mkdir_mode(gdrdrv_proc_dir_name, gdrdrv_proc_dir_mode, NULL);
    if (!gdrdrv_proc_dir_entry) {
        status = EINVAL;
        goto out;
    }

    entry = proc_create("params", entry_mode, gdrdrv_proc_dir_entry, &gdrdrv_proc_params_proc_ops);
    if (!entry) {
        status = EINVAL;
        goto out;
    }

    if (dbg_enabled) {
        entry_mode = (S_IFREG | S_IRUSR | S_IRGRP);
        entry = proc_create("nv_get_pages_refcount", entry_mode, gdrdrv_proc_dir_entry, &gdrdrv_proc_ngp_refcount_proc_ops);
        if (!entry) {
            status = EINVAL;
            goto out;
        }
    }

out:
    if (status) {
        if (gdrdrv_proc_dir_entry) {
            proc_remove(gdrdrv_proc_dir_entry);
            gdrdrv_proc_dir_entry = NULL;
        }
    }
    return status;
}

static void gdrdrv_procfs_cleanup(void)
{
    if (gdrdrv_proc_dir_entry)
        proc_remove(gdrdrv_proc_dir_entry);

    gdrdrv_proc_dir_entry = NULL;
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

    gdr_msg(KERN_INFO, "loading gdrdrv version %s built for %s NVIDIA driver\n", GDRDRV_VERSION_STRING, GDRDRV_BUILT_FOR_NVIDIA_FLAVOR_STRING);
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
#elif defined(CONFIG_ARM64)
    // Grace-Hopper supports CPU cached mapping. But this feature might be disabled at runtime.
    // gdrdrv_pin_buffer will do the right thing.
    gdrdrv_cpu_can_cache_gpu_mappings = 1;
#endif

    if (gdrdrv_cpu_can_cache_gpu_mappings)
        gdr_msg(KERN_INFO, "The platform may support CPU cached mappings. Decision to use cached mappings is left to the pinning function.\n");

#if defined(CONFIG_ARM64)
    {
        // Some compilers are strict and do not allow us to declare a variable
        // in the for statement.
        int i;
        for (i = 0; i < ARRAY_SIZE(GDRDRV_BF3_PCI_ROOT_DEV_DEVICE_ID); ++i)
        {
            struct pci_dev *pdev = pci_get_device(GDRDRV_BF3_PCI_ROOT_DEV_VENDOR_ID, GDRDRV_BF3_PCI_ROOT_DEV_DEVICE_ID[i], NULL);
            if (pdev) {
                pci_dev_put(pdev);
                gdrdrv_cpu_must_use_device_mapping = 1;
                break;
            }
        }
    }
#endif

    if (gdrdrv_cpu_must_use_device_mapping)
        gdr_msg(KERN_INFO, "enabling use of CPU device mappings\n");

    if (gdr_use_persistent_mapping())
        gdr_msg(KERN_INFO, "Persistent mapping will be used\n");

    gdrdrv_procfs_init();

    return 0;
}

//-----------------------------------------------------------------------------

static void __exit gdrdrv_cleanup(void)
{
    int64_t last_nv_get_pages_refcount;
    gdr_msg(KERN_INFO, "unregistering major number %d\n", gdrdrv_major);

    gdrdrv_procfs_cleanup();

    /* cleanup_module is never called if registering failed */
    unregister_chrdev(gdrdrv_major, DEVNAME);

    last_nv_get_pages_refcount = atomic64_read(&gdrdrv_nv_get_pages_refcount);
    if (dbg_enabled)
        WARN_ON(0 != last_nv_get_pages_refcount);
    else
        BUG_ON(0 != last_nv_get_pages_refcount);
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
