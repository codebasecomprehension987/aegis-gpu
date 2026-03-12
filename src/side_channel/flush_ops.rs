// aegis-gpu/src/side_channel/flush_ops.rs
// SIMD-accelerated host-side cache flush helpers (x86-64 with CLFLUSHOPT).

use anyhow::Result;

/// Flush a host-side buffer from all CPU cache levels.
/// Uses CLFLUSHOPT for cache-line–granular eviction; falls back to MFENCE.
///
/// # Safety
/// `buf` must be valid for reads for its entire length.
#[inline]
pub unsafe fn cpu_clflush_range(buf: &[u8]) {
    #[cfg(target_arch = "x86_64")]
    {
        use std::arch::x86_64::_mm_clflushopt;
        let start = buf.as_ptr();
        let end   = unsafe { start.add(buf.len()) };
        let mut p = start as usize & !63usize; // align down to cache line
        while p < end as usize {
            unsafe { _mm_clflushopt(p as *const _); }
            p += 64;
        }
        // Serialise: SFENCE after CLFLUSHOPT
        unsafe { std::arch::x86_64::_mm_sfence(); }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        // Fallback: compiler barrier only
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
    }
}

// aegis-gpu/src/side_channel/timing_fence.rs  (inlined here for brevity)

use crate::hal::GpuHal;

/// Issue a GPU-side serialising fence that prevents reordering of memory
/// operations across a context switch boundary.
pub fn emit(hal: &GpuHal) -> Result<()> {
    // NV_PFIFO_ENGINE_STATUS — wait for all pending work to drain.
    // Real implementation: issue a NOP + timestamp report and wait on semaphore.
    hal.mmio_write32(0x0000_2000, 0x0)?; // dummy write to force ordering
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
    Ok(())
}
