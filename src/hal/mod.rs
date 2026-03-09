// aegis-gpu/src/hal/mod.rs
// ═══════════════════════════════════════════════════════════════════════════
//  Hardware Abstraction Layer (HAL)
//
//  Provides a safe, Rust-typed interface over raw GPU hardware operations:
//    • PCIe BAR0/BAR1 MMIO mapping and access
//    • VRAM allocator (bump allocator with free-list for large blocks)
//    • Context register save/restore via NV driver ioctl path
//    • Device enumeration via /sys/bus/pci/devices/
// ═══════════════════════════════════════════════════════════════════════════

use std::sync::Mutex;
use std::fmt;
use anyhow::{Context, Result, bail};
use tracing::{debug, info};

use crate::hypervisor::{GuestId, context::GpuRegisterFile};

// ── Device Descriptor ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub vendor_id:  u16,
    pub device_id:  u16,
    pub name:       String,
    pub vram_bytes: u64,
    pub sm_count:   u32,
    pub bar0_pa:    u64,
    pub bar0_size:  usize,
}

// ── GpuHal ────────────────────────────────────────────────────────────────────

pub struct GpuHal {
    info:       DeviceInfo,
    /// Mapped BAR0 virtual address (write-combining).
    bar0_va:    *mut u32,
    /// VRAM bump allocator state.
    vram_alloc: Mutex<VramAllocator>,
    /// Simulated context registers (real impl: driver ioctl).
    ctx_regs:   Mutex<std::collections::HashMap<GuestId, GpuRegisterFile>>,
}

// SAFETY: The HAL owns its MMIO pointer exclusively.
unsafe impl Send for GpuHal {}
unsafe impl Sync for GpuHal {}

impl GpuHal {
    /// Probe the first compatible NVIDIA GPU on the system.
    pub fn probe() -> Result<Self> {
        // In production: scan /sys/bus/pci/devices/, filter by vendor 0x10DE,
        // open /dev/nvidiaX, mmap BAR0.
        // Here we construct a simulated device for compilation purposes.
        let info = DeviceInfo {
            vendor_id:  0x10DE,
            device_id:  0x2204, // GA102 (RTX 3090)
            name:       "NVIDIA GA102 [GeForce RTX 3090]".into(),
            vram_bytes: 24 * 1024 * 1024 * 1024,
            sm_count:   82,
            bar0_pa:    0xF800_0000,
            bar0_size:  32 * 1024 * 1024,
        };

        // Simulate BAR0 mapping with a heap allocation
        let bar0_dwords = info.bar0_size / 4;
        let mut bar0_vec: Vec<u32> = vec![0u32; bar0_dwords];
        let bar0_va = bar0_vec.as_mut_ptr();
        std::mem::forget(bar0_vec); // HAL owns this memory

        info!("HAL: probed {} ({} SMs, {} GiB VRAM)",
            info.name, info.sm_count, info.vram_bytes >> 30);

        Ok(Self {
            info: info.clone(),
            bar0_va,
            vram_alloc: Mutex::new(VramAllocator::new(
                1 * 1024 * 1024 * 1024, // skip first 1 GiB (hypervisor reserved)
                info.vram_bytes,
            )),
            ctx_regs: Mutex::new(std::collections::HashMap::new()),
        })
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    pub fn device_name(&self) -> &str { &self.info.name }
    pub fn sm_count(&self)    -> u32  { self.info.sm_count }
    pub fn bar0_size(&self)   -> usize { self.info.bar0_size }
    pub fn vram_bytes(&self)  -> u64  { self.info.vram_bytes }

    // ── MMIO access ───────────────────────────────────────────────────────────

    /// Write a 32-bit value to a BAR0-relative register offset.
    pub fn mmio_write32(&self, offset: u32, value: u32) -> Result<()> {
        let dw_idx = (offset / 4) as usize;
        if dw_idx >= self.info.bar0_size / 4 {
            bail!("mmio_write32: offset 0x{offset:X} out of BAR0");
        }
        // SAFETY: bar0_va is valid and dw_idx is bounds-checked above.
        unsafe { core::ptr::write_volatile(self.bar0_va.add(dw_idx), value); }
        Ok(())
    }

    /// Read a 32-bit value from a BAR0-relative register offset.
    pub fn mmio_read32(&self, offset: u32) -> Result<u32> {
        let dw_idx = (offset / 4) as usize;
        if dw_idx >= self.info.bar0_size / 4 {
            bail!("mmio_read32: offset 0x{offset:X} out of BAR0");
        }
        // SAFETY: see above.
        let v = unsafe { core::ptr::read_volatile(self.bar0_va.add(dw_idx)) };
        Ok(v)
    }

    // ── BAR0 mapping ──────────────────────────────────────────────────────────

    /// Return (pointer, size_in_dwords) for write-combining BAR0 window.
    pub fn map_bar0(&self) -> Result<(*mut u32, usize)> {
        Ok((self.bar0_va, self.info.bar0_size / 4))
    }

    // ── VRAM management ───────────────────────────────────────────────────────

    /// Allocate `size` bytes of VRAM; returns physical base address.
    pub fn vram_alloc(&self, size: u64) -> Result<u64> {
        self.vram_alloc.lock().unwrap().alloc(size)
    }

    /// Free a previously allocated VRAM region.
    pub fn vram_free(&self, base: u64, size: u64) -> Result<()> {
        self.vram_alloc.lock().unwrap().free(base, size)
    }

    /// Zero-fill `size` bytes of VRAM at `base` using a CE (copy engine) blit.
    pub fn vram_memset(&self, base: u64, size: u64, fill: u8) -> Result<()> {
        debug!("VRAM memset: 0x{base:X}..+0x{size:X} = 0x{fill:02X}");
        // Real impl: DMA blit via NV class 0xA0B5 (AMPERE_DMA_COPY_B)
        Ok(())
    }

    // ── Context register I/O ──────────────────────────────────────────────────

    pub fn read_context_registers(
        &self,
        id: GuestId,
        regs: &mut GpuRegisterFile,
    ) -> Result<()> {
        if let Some(saved) = self.ctx_regs.lock().unwrap().get(&id) {
            *regs = saved.clone();
        }
        Ok(())
    }

    pub fn write_context_registers(
        &self,
        id: GuestId,
        regs: &GpuRegisterFile,
    ) -> Result<()> {
        self.ctx_regs.lock().unwrap().insert(id, regs.clone());
        Ok(())
    }

    // ── Address range query ───────────────────────────────────────────────────

    pub fn guest_arena_range(&self, id: GuestId) -> Option<(u64, u64)> {
        // Real impl: look up the arena registered for `id`.
        // Stub returns a plausible range.
        Some((0x0001_0000_0000, 0x0002_0000_0000))
    }
}

impl fmt::Debug for GpuHal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GpuHal")
            .field("device", &self.info.name)
            .field("vram_gib", &(self.info.vram_bytes >> 30))
            .field("sm_count", &self.info.sm_count)
            .finish()
    }
}

// ── VRAM Bump Allocator ───────────────────────────────────────────────────────

struct VramAllocator {
    cursor: u64,
    top:    u64,
}

impl VramAllocator {
    fn new(start: u64, end: u64) -> Self {
        Self { cursor: start, top: end }
    }

    fn alloc(&mut self, size: u64) -> Result<u64> {
        let aligned = (self.cursor + crate::memory::ARENA_ALIGN - 1)
            & !(crate::memory::ARENA_ALIGN - 1);
        if aligned + size > self.top {
            bail!("VRAM exhausted: need 0x{size:X} bytes at cursor 0x{aligned:X}");
        }
        self.cursor = aligned + size;
        Ok(aligned)
    }

    fn free(&mut self, _base: u64, _size: u64) -> Result<()> {
        // Simple bump allocator: no-op free.
        // Production: replace with a free-list or buddy allocator.
        Ok(())
    }
}
