// aegis-gpu/src/side_channel/aegis_scrub.cu
// ═══════════════════════════════════════════════════════════════════════════
//  Aegis GPU Scrub Kernel
//
//  Compiled to a fatbin and embedded in the hypervisor binary.
//  Launched as a privileged, non-preemptible compute context.
//
//  Strategy:
//    • One persistent block per SM (gridDim.x == SM count).
//    • Each block writes SCRUB_PATTERN to all shared memory locations.
//    • A final __threadfence_system() ensures global visibility.
//    • The epoch is written to a scratchpad surface for audit.
// ═══════════════════════════════════════════════════════════════════════════

#include <cuda_runtime.h>
#include <stdint.h>

#define SMEM_SIZE_BYTES (48 * 1024)   // 48 KiB — Ampere default shared mem
#define SCRUB_WORDS     (SMEM_SIZE_BYTES / sizeof(uint32_t))

extern "C" __global__ __launch_bounds__(1024, 1)
void aegis_scrub_kernel(
    uint32_t  scrub_pattern,
    uint64_t  epoch,
    uint32_t* audit_surface   // one slot per SM
) {
    // Declare maximum shared memory allocation.
    __shared__ uint32_t smem[SCRUB_WORDS];

    const int tid  = threadIdx.x;
    const int ntid = blockDim.x;

    // ── Phase 1: Fill shared memory with scrub pattern ────────────────────
    // Stride loop ensures all SCRUB_WORDS locations are written regardless
    // of block size.
    for (int i = tid; i < SCRUB_WORDS; i += ntid) {
        smem[i] = scrub_pattern;
    }
    __syncthreads();

    // ── Phase 2: Read back (force cache eviction via repeated access) ──────
    uint32_t acc = 0;
    for (int i = tid; i < SCRUB_WORDS; i += ntid) {
        acc ^= smem[i];
    }
    __syncthreads();

    // ── Phase 3: Zero shared memory ───────────────────────────────────────
    for (int i = tid; i < SCRUB_WORDS; i += ntid) {
        smem[i] = 0;
    }
    __syncthreads();

    // ── Phase 4: Audit write + system-wide fence ──────────────────────────
    if (tid == 0) {
        // Prevent dead-code elimination of Phase 2 accumulator
        audit_surface[blockIdx.x] = (uint32_t)(epoch & 0xFFFFFFFF) ^ acc;
        __threadfence_system();
    }
}

// ── L1 / texture flush helper ─────────────────────────────────────────────
// Launched as a separate kernel to avoid shared memory contention.
extern "C" __global__
void aegis_l1_invalidate_kernel(void) {
    // Writing to a global surface through a non-cached path forces L1
    // eviction on Ampere when combined with cache control modifiers.
    asm volatile("fence.sc.gpu;" ::: "memory");
    asm volatile("membar.gl;" ::: "memory");
}
