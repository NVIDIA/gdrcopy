#!/bin/sh
# Exercise Makefile probe environment without a kernel tree or GPU.
set -eu
repo=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
make_bin=${1:-make}
tmpdir=$(mktemp -d)
trap 'rm -rf "$tmpdir"' EXIT HUP INT TERM
cat > "$tmpdir/probe" <<'PROBE'
#!/bin/sh
printf '%s\n' "${KDIR-unset}" >> "$PROBE_LOG"
echo n
PROBE
chmod +x "$tmpdir/probe"
for feature in HAVE_PROC_OPS HAVE_VM_FLAGS_SET; do
    cp "$tmpdir/probe" "$tmpdir/test_gdrdrv_${feature}.sh"
done
for mode in override fallback; do
    : > "$tmpdir/log"
    if [ "$mode" = override ]; then
        set -- "KDIR=$tmpdir/custom kernel"
        expected="$tmpdir/custom kernel"
    else
        set --
        expected=/lib/modules/test-kernel/build
    fi
    env -u KDIR PROBE_LOG="$tmpdir/log" "$make_bin" --no-print-directory \
        -C "$repo/src/gdrdrv" -n clean KVER=test-kernel \
        "CONF_SCRIPT_DIR=$tmpdir" NVIDIA_SRC_DIR=/nonexistent "$@" >/dev/null 2>&1
    printf '%s\n%s\n' "$expected" "$expected" > "$tmpdir/expected"
    diff -u "$tmpdir/expected" "$tmpdir/log"
done

# Check that each real feature probe passes the override as one make argument.
mkdir "$tmpdir/bin"
cat > "$tmpdir/bin/make" <<'MAKE'
#!/bin/sh
[ "$1" = -C ] && [ "$2" = "$EXPECTED_KDIR" ] && [ "$#" = 4 ]
MAKE
chmod +x "$tmpdir/bin/make"
for feature in HAVE_PROC_OPS HAVE_VM_FLAGS_SET; do
    for mode in override fallback; do
        if [ "$mode" = override ]; then
            expected="$tmpdir/custom kernel"
            result=$(PATH="$tmpdir/bin:$PATH" KDIR="$expected" EXPECTED_KDIR="$expected" \
                sh "$repo/scripts/test_gdrdrv_${feature}.sh" -k test-kernel)
        else
            expected=/lib/modules/test-kernel/build
            result=$(env -u KDIR PATH="$tmpdir/bin:$PATH" EXPECTED_KDIR="$expected" \
                sh "$repo/scripts/test_gdrdrv_${feature}.sh" -k test-kernel)
        fi
        [ "$result" = y ] || { echo "$feature ($mode): expected y, got $result" >&2; exit 1; }
    done
done
echo "KDIR probe tests passed"
