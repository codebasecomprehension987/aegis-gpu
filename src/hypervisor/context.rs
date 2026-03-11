// aegis-gpu/src/hypervisor/context.rs
// Guest execution context — all state that must be saved and restored on
// every preemptive context switch.

use std::time::{Duration, Instant};
use anyhow::Result;
use tracing::{debug, trace};

use crate::hal::GpuHal;
use crate::memory::IsolatedArena;

// ── GuestId ───────────────────────────────────────────────────────────────────

/// Opaque, unforgeable, globally unique guest identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct GuestId(pub(crate) u64);

impl GuestId {
    /// Generate a new unique ID using a thread-local counter.
    pub fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

impl Default for GuestId {
    fn default() -> Self { Self::new() }
}

// ── ContextState ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextState {
    /// Waiting to be scheduled.
    Runnable,
    /// Currently executing on the GPU.
    Running,
    /// Voluntarily suspended, waiting on a host-side event.
    Blocked,
    /// Fatally faulted — will not be rescheduled.
    Faulted,
}

// ── GpuRegisterFile ───────────────────────────────────────────────────────────

/// A snapshot of all hardware state that belongs to a single guest context.
/// Fields mirror the NVIDIA GPU Save/Restore registers documented in the
/// open-source `open-gpu-kernel-modules` (Turing+).
#[derive(Debug, Default, Clone)]
#[repr(C, align(128))] // 128-byte alignment for non-temporal SIMD store/load
pub struct GpuRegisterFile {
    /// GPFIFO Get/Put pointers
    pub gpfifo_get:          u64,
    pub gpfifo_put:          u64,
    /// Compute class handle
    pub compute_class:       u32,
    /// Current shader program counter
    pub shader_pc:           u64,
    /// Shared memory base address (per-CTA)
    pub smem_base:           u64,
    /// Local memory base
    pub lmem_base:           u64,
    /// Warp register file snapshot (pointer into arena)
    pub warp_state_ptr:      u64,
    /// CTA grid dimensions at time of preemption
    pub grid_dim_x:          u32,
    pub grid_dim_y:          u32,
    pub grid_dim_z:          u32,
    /// Fence completion value
    pub fence_val:           u64,
    /// Extended state for fault recovery
    pub fault_addr:          u64,
    pub fault_type:          u32,
    pub _pad:                [u8; 44],
}

// Compile-time size assertion: register file must fit in 2 cache lines.
const _: () = assert!(std::mem::size_of::<GpuRegisterFile>() <= 256);

// ── GuestContext ──────────────────────────────────────────────────────────────

pub struct GuestContext {
    pub id:            GuestId,
    pub state:         ContextState,
    pub priority:      u8,
    /// Isolated VRAM arena — exclusively owned by this guest.
    pub arena:         IsolatedArena,
    /// Saved register file (valid only when state != Running).
    pub regs:          GpuRegisterFile,
    /// Cumulative GPU time used (for fair scheduling).
    pub gpu_time_used: Duration,
    /// Wall-clock time when this context last began executing.
    run_start:         Option<Instant>,
    /// Number of times this context has been preempted.
    pub preempt_count: u64,
}

impl GuestContext {
    pub fn new(id: GuestId, arena: IsolatedArena, priority: u8) -> Self {
        Self {
            id,
            state:         ContextState::Runnable,
            priority:      priority.clamp(0, 15),
            arena,
            regs:          GpuRegisterFile::default(),
            gpu_time_used: Duration::ZERO,
            run_start:     None,
            preempt_count: 0,
        }
    }

    /// Snapshot GPU register state into this context.
    pub fn save(&mut self, hal: &GpuHal) -> Result<()> {
        debug_assert_eq!(self.state, ContextState::Running);

        hal.read_context_registers(self.id, &mut self.regs)?;
        self.state = ContextState::Runnable;

        // Accumulate GPU time
        if let Some(start) = self.run_start.take() {
            self.gpu_time_used += start.elapsed();
        }

        self.preempt_count += 1;
        trace!("Context {:?} saved (preempts={})", self.id, self.preempt_count);
        Ok(())
    }

    /// Restore register state onto the GPU and resume execution.
    pub fn restore(&mut self, hal: &GpuHal) -> Result<()> {
        debug_assert_ne!(self.state, ContextState::Running);

        hal.write_context_registers(self.id, &self.regs)?;
        self.state = ContextState::Running;
        self.run_start = Some(Instant::now());

        trace!("Context {:?} restored", self.id);
        Ok(())
    }

    /// Mark the context as faulted; it will not be rescheduled.
    pub fn fault(&mut self, addr: u64, fault_type: u32) {
        self.regs.fault_addr = addr;
        self.regs.fault_type = fault_type;
        self.state = ContextState::Faulted;
        tracing::error!("Context {:?} FAULTED at 0x{addr:016X} type={fault_type}", self.id);
    }
}
