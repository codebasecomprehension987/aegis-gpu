// aegis-gpu/src/hypervisor/bar_shadow.rs
// ═══════════════════════════════════════════════════════════════════════════
//  PCIe BAR0 Shadowing
//
//  Instead of allowing guest libraries to write directly to the GPU's
//  MMIO aperture (BAR0), Aegis maps a *shadow* copy of BAR0 in its own
//  address space using mmap(MAP_SHARED) over /dev/mem or a UIO device,
//  then interposes on every DWORD write via a write-combining window.
//
//  The shadow region is a ring buffer per guest channel.  The dispatcher
//  loop drains validated packets from the shadow ring and replays them
//  onto the *real* MMIO aperture.
// ═══════════════════════════════════════════════════════════════════════════

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::ptr::NonNull;

use anyhow::{Context, Result, bail};
use tracing::{debug, trace};

use crate::hal::GpuHal;
use super::{GuestId, RING_SIZE_DW, MAX_GUESTS};

// ── Constants ────────────────────────────────────────────────────────────────

/// GPU FIFO method — NV_CHANNELGPFIFO_PREEMPT (Ampere+)
const NV_GPFIFO_PREEMPT_METHOD: u32 = 0x0000_0050;

/// Sentinel placed at the end of every validated packet for tamper detection.
const PACKET_SENTINEL: u32 = 0xAE_61_5AFE;

// ── Data structures ───────────────────────────────────────────────────────────

/// One cache-line–aligned ring for a single guest channel.
/// Layout: [head: u32][tail: u32][pad: 56 bytes][data: RING_SIZE_DW × u32]
#[repr(C, align(64))]
struct GuestRing {
    head:     AtomicU32,
    tail:     AtomicU32,
    _pad:     [u8; 56],
    data:     [AtomicU32; RING_SIZE_DW],
}

impl GuestRing {
    const fn new() -> Self {
        // SAFETY: AtomicU32 has the same representation as u32 = 0.
        // This entire struct is zero-initialised which is correct for an
        // empty ring.
        unsafe { std::mem::zeroed() }
    }

    /// Enqueue `words` into the ring.  Returns Err if the ring is full.
    fn enqueue(&self, words: &[u32]) -> Result<()> {
        let len = words.len() as u32;
        let tail = self.tail.load(Ordering::Acquire);
        let head = self.head.load(Ordering::Acquire);
        let free = RING_SIZE_DW as u32 - (tail.wrapping_sub(head));

        if free < len + 1 {
            bail!("Guest ring full (free={free}, needed={})", len + 1);
        }

        for (i, &w) in words.iter().enumerate() {
            let idx = ((tail + i as u32) as usize) % RING_SIZE_DW;
            self.data[idx].store(w, Ordering::Relaxed);
        }
        // Sentinel after payload
        let sentinel_idx = ((tail + len) as usize) % RING_SIZE_DW;
        self.data[sentinel_idx].store(PACKET_SENTINEL, Ordering::Relaxed);

        // Publish
        self.tail.store(tail.wrapping_add(len + 1), Ordering::Release);
        trace!("Ring enqueue: {len} DWORDs, new_tail={}", tail + len + 1);
        Ok(())
    }

    /// Drain all pending words into `out`.  Returns count drained.
    fn drain(&self, out: &mut Vec<u32>) -> usize {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        let count = tail.wrapping_sub(head) as usize;

        for i in 0..count {
            let idx = (head as usize + i) % RING_SIZE_DW;
            out.push(self.data[idx].load(Ordering::Relaxed));
        }
        self.head.store(tail, Ordering::Release);
        count
    }
}

// ── BarShadow ─────────────────────────────────────────────────────────────────

pub struct BarShadow {
    hal:          Arc<GpuHal>,
    /// Per-guest software rings — heap allocated to avoid 4 MB stack usage.
    rings:        Box<[GuestRing; MAX_GUESTS]>,
    /// Maps GuestId → ring slot index.
    slot_map:     std::sync::Mutex<[Option<GuestId>; MAX_GUESTS]>,
    /// Currently active channel index written to real BAR0.
    active_slot:  AtomicU32,
    /// Monotonic write counter for replay-guard.
    write_seq:    AtomicU64,
    /// Pointer to the real MMIO window (BAR0) — write-combining mapped.
    /// SAFETY invariant: only written by the dispatcher thread (single writer).
    mmio_ptr:     NonNull<u32>,
    mmio_size_dw: usize,
}

// SAFETY: BarShadow is Send+Sync because:
//   - `rings` is accessed only through &self with atomic ops.
//   - `mmio_ptr` is written only by the single dispatcher thread; all other
//     accesses are reads behind the active_slot atomic gate.
unsafe impl Send for BarShadow {}
unsafe impl Sync for BarShadow {}

impl BarShadow {
    /// Map BAR0 and construct shadow rings.
    pub fn map(hal: Arc<GpuHal>) -> Result<Self> {
        let (mmio_ptr, mmio_size_dw) = hal.map_bar0()
            .context("HAL: BAR0 map failed")?;

        // SAFETY: hal.map_bar0() guarantees the pointer is non-null and
        // correctly aligned to u32, backed by write-combining MMIO memory.
        let mmio_ptr = NonNull::new(mmio_ptr)
            .context("HAL returned null BAR0 pointer")?;

        // SAFETY: zeroed() is valid for the AtomicU32-based ring struct.
        let rings: Box<[GuestRing; MAX_GUESTS]> = {
            let mut v: Vec<GuestRing> = Vec::with_capacity(MAX_GUESTS);
            for _ in 0..MAX_GUESTS { v.push(GuestRing::new()); }
            v.into_boxed_slice()
                .try_into()
                .map_err(|_| anyhow::anyhow!("ring init failed"))?
        };

        Ok(Self {
            hal,
            rings,
            slot_map:    std::sync::Mutex::new([None; MAX_GUESTS]),
            active_slot: AtomicU32::new(u32::MAX),
            write_seq:   AtomicU64::new(0),
            mmio_ptr,
            mmio_size_dw,
        })
    }

    /// Enqueue a validated packet for `guest_id`.
    pub fn enqueue(&self, guest_id: GuestId, packet: &crate::hypervisor::packet_validator::ValidatedPacket) -> Result<()> {
        let slot = self.slot_for(guest_id)
            .context("Guest has no allocated ring slot")?;
        self.rings[slot].enqueue(packet.dwords())?;
        Ok(())
    }

    /// Switch the active hardware channel to `guest_id`.
    /// Called after the side-channel flush, before restoring guest context.
    pub fn switch_active_channel(&self, guest_id: GuestId) -> Result<()> {
        let slot = self.slot_for(guest_id)
            .context("switch_active_channel: unknown guest")?;

        debug!("BAR shadow: activating slot {slot} for guest {:?}", guest_id);
        self.active_slot.store(slot as u32, Ordering::SeqCst);

        // Replay pending ring entries onto real MMIO for the newly active guest.
        self.flush_to_mmio(slot)
    }

    // ── Internal ─────────────────────────────────────────────────────────────

    fn slot_for(&self, id: GuestId) -> Option<usize> {
        let map = self.slot_map.lock().unwrap();
        map.iter().position(|s| *s == Some(id))
    }

    /// Drain the ring for `slot` and replay onto real MMIO BAR0.
    ///
    /// # Safety
    ///
    /// This function performs volatile MMIO writes.  It must only be called
    /// from the single dispatcher thread that owns the MMIO window.
    fn flush_to_mmio(&self, slot: usize) -> Result<()> {
        let mut words = Vec::with_capacity(RING_SIZE_DW);
        let count = self.rings[slot].drain(&mut words);

        if count == 0 {
            return Ok(());
        }

        // Strip sentinel words before writing to hardware
        let payload: Vec<u32> = words
            .into_iter()
            .filter(|&w| w != PACKET_SENTINEL)
            .collect();

        for (i, &dw) in payload.iter().enumerate() {
            // Bounds guard: never write past BAR0
            if i >= self.mmio_size_dw {
                bail!("MMIO overrun: attempt to write DW[{i}] beyond BAR0 ({} DW)", self.mmio_size_dw);
            }
            // SAFETY:
            // - `mmio_ptr` is valid and write-combining mapped (constructor invariant).
            // - `i < mmio_size_dw` enforced above.
            // - Single-writer invariant upheld (only dispatcher calls this).
            unsafe {
                core::ptr::write_volatile(self.mmio_ptr.as_ptr().add(i), dw);
            }
            self.write_seq.fetch_add(1, Ordering::Relaxed);
        }

        // Memory barrier: ensure all writes reach the device before returning.
        std::sync::atomic::fence(Ordering::SeqCst);

        debug!("BAR flush: {count} DWORDs replayed to MMIO (slot={slot})");
        Ok(())
    }
}

impl Drop for BarShadow {
    fn drop(&mut self) {
        // Issue a PREEMPT method to halt any in-flight work before unmapping.
        let active = self.active_slot.load(Ordering::SeqCst) as usize;
        if active < MAX_GUESTS {
            unsafe {
                core::ptr::write_volatile(
                    self.mmio_ptr.as_ptr(),
                    NV_GPFIFO_PREEMPT_METHOD,
                );
            }
        }
        // HAL is responsible for unmapping BAR0 on drop.
        debug!("BarShadow dropped — PREEMPT issued");
    }
}
