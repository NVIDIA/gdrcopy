typedef struct gdr_memh_t { 
    uint32_t handle;
    LIST_ENTRY(gdr_memh_t) entries;
    unsigned mapped:1;
    unsigned wc_mapping:1;
} gdr_memh_t;

struct gdr {
    int fd;
    LIST_HEAD(memh_list, gdr_memh_t) memhs;
};

