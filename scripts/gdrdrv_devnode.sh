#!/bin/sh

set -e

NODE=/dev/gdrdrv

set_perms()
{
    chmod 0666 "${NODE}" 2>/dev/null || true
}

create_node()
{
    major=$(awk '$2 == "gdrdrv" { print $1; exit }' /proc/devices)
    if [ -z "${major}" ]; then
        return 0
    fi

    i=0
    while [ "${i}" -lt 20 ]; do
        if [ -e "${NODE}" ]; then
            break
        fi
        sleep 0.1
        i=$((i + 1))
    done

    need_mknod=1
    if [ -e "${NODE}" ]; then
        cur_hex=$(stat -c '%t' "${NODE}" 2>/dev/null || true)
        if [ -n "${cur_hex}" ]; then
            cur=$(printf '%d' "0x${cur_hex}" 2>/dev/null || true)
            if [ "${cur}" = "${major}" ]; then
                need_mknod=0
            fi
        fi
        if [ "${need_mknod}" -eq 1 ]; then
            if ! rm -f "${NODE}"; then
                echo "ERROR: failed to remove stale ${NODE}" >&2
                return 1
            fi
        fi
    fi

    if [ "${need_mknod}" -eq 1 ]; then
        if ! mknod "${NODE}" c "${major}" 0; then
            echo "ERROR: failed to create ${NODE} (major ${major})" >&2
            return 1
        fi
    fi
    set_perms
}

remove_node()
{
    if ! rm -f "${NODE}"; then
        echo "ERROR: failed to remove ${NODE}" >&2
        return 1
    fi
}

case "$1" in
    create)
        create_node
        ;;
    remove)
        remove_node
        ;;
    *)
        echo "Usage: $0 {create|remove}" >&2
        exit 1
        ;;
esac
