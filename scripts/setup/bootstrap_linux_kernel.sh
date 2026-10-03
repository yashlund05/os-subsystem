#!/usr/bin/env bash
# bootstrap_linux_kernel.sh — Phase 1, Week 1
# Verifies that the running kernel is ≥ 6.12 and has sched_ext BPF headers
# available. Exits non-zero if any check fails.
#
# Usage:
#   bash scripts/setup/bootstrap_linux_kernel.sh
#
# Requirements:
#   - Linux x86_64 or ARM64
#   - libbpf-dev / libbpf-devel installed
#   - Kernel headers: linux-headers-$(uname -r) or equivalent

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

REQUIRED_MAJOR=6
REQUIRED_MINOR=12

# ---------------------------------------------------------------------------
# 1. Kernel version check
# ---------------------------------------------------------------------------
info "Checking kernel version …"
KERNEL_VERSION="$(uname -r)"
KERNEL_MAJOR="$(echo "${KERNEL_VERSION}" | cut -d. -f1)"
KERNEL_MINOR="$(echo "${KERNEL_VERSION}" | cut -d. -f2 | cut -d- -f1)"

if [[ "${KERNEL_MAJOR}" -lt "${REQUIRED_MAJOR}" ]] || \
   { [[ "${KERNEL_MAJOR}" -eq "${REQUIRED_MAJOR}" ]] && \
     [[ "${KERNEL_MINOR}" -lt "${REQUIRED_MINOR}" ]]; }; then
    error "Kernel ${KERNEL_VERSION} is too old. " \
          "NeuroOS-Lite requires Linux >= ${REQUIRED_MAJOR}.${REQUIRED_MINOR}."
    error "Please upgrade your kernel before continuing."
    exit 1
fi
info "Kernel ${KERNEL_VERSION} ✓ (>= ${REQUIRED_MAJOR}.${REQUIRED_MINOR} required)"

# ---------------------------------------------------------------------------
# 2. sched_ext BPF header check
# ---------------------------------------------------------------------------
info "Checking for sched_ext BPF headers …"
SCHED_EXT_HEADER="/usr/include/linux/sched_ext.h"
SCHED_EXT_BPF_HEADER="/usr/include/bpf/sched_ext.h"
FOUND_HEADER=""

for h in "${SCHED_EXT_HEADER}" "${SCHED_EXT_BPF_HEADER}"; do
    if [[ -f "${h}" ]]; then
        FOUND_HEADER="${h}"
        break
    fi
done

if [[ -z "${FOUND_HEADER}" ]]; then
    warn "sched_ext header not found at standard paths."
    warn "Searching via find (this may take a moment) …"
    FOUND_HEADER="$(find /usr/include /usr/src -name "sched_ext.h" 2>/dev/null | head -1 || true)"
fi

if [[ -z "${FOUND_HEADER}" ]]; then
    error "sched_ext BPF header not found."
    error "Install with: apt install linux-headers-\$(uname -r) libbpf-dev"
    error "  or (Fedora/RHEL): dnf install kernel-headers libbpf-devel"
    exit 1
fi
info "sched_ext header found: ${FOUND_HEADER} ✓"

# ---------------------------------------------------------------------------
# 3. libbpf check
# ---------------------------------------------------------------------------
info "Checking for libbpf …"
if ! ldconfig -p 2>/dev/null | grep -q libbpf; then
    if ! pkg-config --exists libbpf 2>/dev/null; then
        error "libbpf not found. Install with: apt install libbpf-dev"
        exit 1
    fi
fi
info "libbpf ✓"

# ---------------------------------------------------------------------------
# 4. Architecture check
# ---------------------------------------------------------------------------
info "Checking CPU architecture …"
ARCH="$(uname -m)"
case "${ARCH}" in
    x86_64)
        info "Architecture x86_64 ✓ — AVX2 SIMD path will be used."
        ;;
    aarch64)
        info "Architecture aarch64 ✓ — ARM NEON SIMD path will be used."
        ;;
    *)
        warn "Unsupported architecture: ${ARCH}. " \
             "Generic C path will be compiled (no SIMD acceleration)."
        ;;
esac

# ---------------------------------------------------------------------------
# 5. Summary
# ---------------------------------------------------------------------------
echo
info "=== Bootstrap verification PASSED ==="
info "Kernel:       ${KERNEL_VERSION}"
info "sched_ext:    ${FOUND_HEADER}"
info "Architecture: ${ARCH}"
echo
info "Next step: run 'bash scripts/setup/install_python_deps.sh'"
