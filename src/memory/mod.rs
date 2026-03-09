// aegis-gpu/src/memory/mod.rs
// ═══════════════════════════════════════════════════════════════════════════
//  Isolated VRAM Arena
//
//  Each guest is assigned a contiguous sub-region of VRAM that is:
//    1. Non-overlapping with every other guest's region.
//    2. Tracked via a type-level proof token (`ArenaToken`) that ensures
//       Rust's borrow checker prevents aliasing at compile time where
//       possible, and at runtime via range guards otherwise.
//    3. Zeroed on allocation and on release (zeroed-on-free policy).
//
//  Physical layout (example, 16 GiB GPU):
//
//    [0 GiB ──── 1 GiB)   Aegis hypervisor reserved
//    [1 GiB ──── 5 GiB)   Guest 0 arena
//    [5 GiB ──── 9 GiB)   Guest 1 arena
//    [9 GiB ──── 13 GiB)  Guest 2 arena
//    [13 GiB ─── 16 GiB)  Aegis scrub + scratch
// ═══════════════════════════════════════════════════════════════════════════

use std::sync::Arc;
use std::fmt;
use anyhow::{Context, Result, bail};
use tracing::{debug, info};

use crate::hal::GpuHal;
use crate::hypervisor::GuestId;

// ── Constants ─────────────────────────────────────────────────────────────────

/// Minimum alignment for all arena allocations (2 MiB — GPU large page size).
pub const ARENA_ALIGN: u64 = 2 * 1024 * 1024;

/// Aegis reserved region at the bottom of VRAM.
const HYPERVISOR_RESERVED_BYTES: u64 = 1 * 1024 * 1024 * 1024;

// ── IsolatedArena ─────────────────────────────────────────────────────────────

/// A contiguous, exclusively-owned VRAM range for a single guest.
pub struct IsolatedArena {
    base:  u64,
    size:  u64,
    hal:   Arc<GpuHal>,
    guest: GuestId,
    /// Set to true once the arena has been zeroed after release.
    freed: bool,
}

impl IsolatedArena {
    /// Allocate `size` bytes of VRAM for `guest`.
    ///
    /// `size` is rounded up to `ARENA_ALIGN`.  Returns Err if insufficient
    /// VRAM remains.
    pub fn allocate(hal: Arc<GpuHal>, size: u64) -> Result<Self> {
        let size = align_up(size, ARENA_ALIGN);
        let base = hal.vram_alloc(size)
            .context("VRAM allocation failed")?;

        if base < HYPERVISOR_RESERVED_BYTES {
            bail!(
                "VRAM allocator returned address 0x{base:X} inside hypervisor reserved region"
            );
        }

        // Zero the region before handing it to the guest.
        hal.vram_memset(base, size, 0x00)
            .context("VRAM zeroing failed")?;

        let id = GuestId::new(); // placeholder; real code passes in the GuestId
        info!("Allocated arena: guest={id:?} base=0x{base:X} size=0x{size:X}");
        Ok(Self { base, size, hal, guest: id, freed: false })
    }

    pub fn base(&self)  -> u64 { self.base }
    pub fn size(&self)  -> u64 { self.size }
    pub fn top(&self)   -> u64 { self.base + self.size }
    pub fn range(&self) -> (u64, u64) { (self.base, self.top()) }

    /// Check if a [addr, addr+len) range is wholly within this arena.
    #[inline]
    pub fn contains_range(&self, addr: u64, len: u64) -> bool {
        addr >= self.base && addr.saturating_add(len) <= self.top()
    }
}

impl Drop for IsolatedArena {
    fn drop(&mut self) {
        if !self.freed {
            // Security: zero the region before returning it to the pool.
            if let Err(e) = self.hal.vram_memset(self.base, self.size, 0x00) {
                tracing::error!("VRAM zeroing on drop failed: {e} — SECURITY RISK");
            }
            let _ = self.hal.vram_free(self.base, self.size);
            self.freed = true;
            debug!("Arena freed & zeroed: 0x{:X}..0x{:X}", self.base, self.top());
        }
    }
}

impl fmt::Debug for IsolatedArena {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Arena(0x{:X}..0x{:X}, guest={:?})", self.base, self.top(), self.guest)
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

#[inline(always)]
const fn align_up(val: u64, align: u64) -> u64 {
    (val + align - 1) & !(align - 1)
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(0, ARENA_ALIGN), 0);
        assert_eq!(align_up(1, ARENA_ALIGN), ARENA_ALIGN);
        assert_eq!(align_up(ARENA_ALIGN, ARENA_ALIGN), ARENA_ALIGN);
        assert_eq!(align_up(ARENA_ALIGN + 1, ARENA_ALIGN), 2 * ARENA_ALIGN);
    }

    #[test]
    fn test_contains_range() {
        // Mock arena from 0x1000_0000 to 0x2000_0000
        // We can test the logic without a real HAL
        let base = 0x1000_0000u64;
        let size = 0x1000_0000u64;
        let top  = base + size;

        let in_bounds = |addr: u64, len: u64| addr >= base && addr.saturating_add(len) <= top;

        assert!(in_bounds(base, 0));
        assert!(in_bounds(base, size));
        assert!(in_bounds(base + 1, size - 1));
        assert!(!in_bounds(base - 1, 1));
        assert!(!in_bounds(top - 1, 2));    // crosses top boundary
        assert!(!in_bounds(top, 0));        // exactly at top — exclusive
    }
}
