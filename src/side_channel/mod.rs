// aegis-gpu/src/side_channel/mod.rs
// ═══════════════════════════════════════════════════════════════════════════
//  Side-Channel Mitigator
//
//  Performs the security-critical flush sequence between GPU context
//  switches.  The goal is to eliminate microarchitectural state that could
//  allow Tenant B to infer anything about Tenant A's computation via:
//
//    • L2 Cache timing attacks      — flush + fill with scrub pattern
//    • Shared memory residue        — zero-fill via an Aegis scrub kernel
//    • Register file leakage        — GPUs write random values on alloc,
//                                     but we additionally request a 
//                                     register-file reset via the driver API
//    • Texture/L1 state             — invalidate via NV_GRAPH_INVALID_*
//
//  Hardware flush mechanism (Ampere+):
//    Write NV_PLTCG_LTC_FLUSH_L2_SECTOR to LTC flush registers via MMIO,
//    then poll NV_PLTCG_LTC_FLUSH_DONE until the bit is set.
//    If polling exceeds FLUSH_TIMEOUT_US, emit a warning and proceed anyway
//    (availability > perfect security in steady state; faults logged).
// ═══════════════════════════════════════════════════════════════════════════

pub mod flush_ops;
pub mod scrub_kernel;
pub mod timing_fence;

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use anyhow::{Context, Result};
use tracing::{debug, warn, instrument};

use crate::hal::GpuHal;

// ── Constants ─────────────────────────────────────────────────────────────────

/// MMIO offset for the LTC (L2 TLB Cache) flush control register.
const NV_PLTCG_LTC_FLUSH_L2: u32   = 0x0017_E200;
const NV_PLTCG_LTC_FLUSH_DONE: u32 = 0x0017_E204;

/// How long to wait for the hardware flush to complete.
const FLUSH_TIMEOUT_US: u64 = 5_000; // 5 ms

/// Scrub pattern written over shared memory to prevent data remanence.
const SCRUB_PATTERN: u32 = 0xDEAD_C0DE;

// ── Stats ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Default)]
pub struct FlushStats {
    pub total_flushes:   AtomicU64,
    pub timeout_flushes: AtomicU64,
    pub total_flush_us:  AtomicU64,
}

// ── SideChannelMitigator ──────────────────────────────────────────────────────

pub struct SideChannelMitigator {
    hal:   Arc<GpuHal>,
    stats: Arc<FlushStats>,
}

impl SideChannelMitigator {
    pub fn new(hal: Arc<GpuHal>) -> Result<Self> {
        // Verify scrub kernel is compiled and present on device.
        scrub_kernel::verify_present(&hal)
            .context("Scrub kernel not found on device")?;

        Ok(Self {
            hal,
            stats: Arc::new(FlushStats::default()),
        })
    }

    /// Execute the full flush pipeline.
    ///
    /// Must complete before the next guest's context is restored.
    /// This is the performance-critical hot path — every µs counts.
    #[instrument(skip(self), fields(epoch))]
    pub async fn full_flush(&self, epoch: u64) -> Result<()> {
        tracing::Span::current().record("epoch", epoch);
        let t0 = Instant::now();

        // ── 1. L2 / LTC cache flush ──────────────────────────────────────────
        self.flush_l2_cache()
            .context("L2 flush failed")?;

        // ── 2. Shared memory scrub (GPU-side kernel) ─────────────────────────
        // Launch the Aegis scrub kernel which zeroes all shared memory banks
        // across every active SM.  This runs as a hypervisor-privileged context
        // that cannot be preempted by guest code.
        scrub_kernel::launch_scrub(&self.hal, SCRUB_PATTERN, epoch)
            .context("Scrub kernel launch failed")?;

        // ── 3. L1 / texture cache invalidate ─────────────────────────────────
        self.invalidate_l1_texture_caches()
            .context("L1/texture invalidate failed")?;

        // ── 4. Timing fence — prevent speculative reads across switch ─────────
        timing_fence::emit(&self.hal)?;

        let elapsed_us = t0.elapsed().as_micros() as u64;
        self.stats.total_flushes.fetch_add(1, Ordering::Relaxed);
        self.stats.total_flush_us.fetch_add(elapsed_us, Ordering::Relaxed);

        debug!("Full flush complete in {elapsed_us}µs (epoch={epoch})");
        Ok(())
    }

    pub fn stats(&self) -> &FlushStats { &self.stats }

    // ── Private ───────────────────────────────────────────────────────────────

    /// Hardware L2 flush via LTC MMIO registers.
    fn flush_l2_cache(&self) -> Result<()> {
        // Trigger flush
        self.hal.mmio_write32(NV_PLTCG_LTC_FLUSH_L2, 0x1)?;

        // Poll for completion with timeout
        let deadline = Instant::now() + Duration::from_micros(FLUSH_TIMEOUT_US);
        loop {
            let done = self.hal.mmio_read32(NV_PLTCG_LTC_FLUSH_DONE)?;
            if done & 0x1 != 0 {
                // Clear the done bit
                self.hal.mmio_write32(NV_PLTCG_LTC_FLUSH_DONE, 0x1)?;
                return Ok(());
            }
            if Instant::now() >= deadline {
                warn!("L2 flush timed out after {}µs — proceeding anyway", FLUSH_TIMEOUT_US);
                self.stats.timeout_flushes.fetch_add(1, Ordering::Relaxed);
                return Ok(()); // degraded but non-fatal
            }
            // Tight spin — we cannot afford to sleep here
            core::hint::spin_loop();
        }
    }

    fn invalidate_l1_texture_caches(&self) -> Result<()> {
        // NV_PGRAPH_INVALID_* sequence (class-specific).
        // Write 0x1 to each SM's L1 invalidate register.
        for sm_idx in 0..self.hal.sm_count() {
            let reg = 0x0050_4400 + (sm_idx as u32) * 0x8000;
            self.hal.mmio_write32(reg, 0x1)?;
        }
        // Full memory barrier before continuing
        std::sync::atomic::fence(Ordering::SeqCst);
        Ok(())
    }
}
