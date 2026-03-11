// aegis-gpu/src/hypervisor/mod.rs
// ═══════════════════════════════════════════════════════════════════════════
//  Aegis-GPU Microkernel  —  Hypervisor Core
//
//  Responsibilities
//  ─────────────────
//  1. Owns the canonical list of GuestContext handles (one per tenant).
//  2. Drives the PCIe BAR shadow loop: intercepts every PUSHBUF/GPFIFO
//     packet before it reaches real MMIO, validates it, then either
//     forwards or rejects it.
//  3. Delegates scheduling decisions to the FairScheduler.
//  4. Calls SideChannelMitigator between every context switch.
// ═══════════════════════════════════════════════════════════════════════════

pub mod bar_shadow;
pub mod context;
pub mod packet_validator;

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use tracing::{debug, info, warn, instrument};
use anyhow::{Context, Result, bail};

use crate::hal::GpuHal;
use crate::ipc::IpcServer;
use crate::memory::IsolatedArena;
use crate::scheduler::{FairScheduler, ScheduleDecision};
use crate::side_channel::SideChannelMitigator;
use crate::ptx_rewriter::PtxRewriter;

pub use context::{GuestContext, GuestId, ContextState};
pub use bar_shadow::BarShadow;
pub use packet_validator::PacketValidator;

// ── Types ────────────────────────────────────────────────────────────────────

/// Maximum concurrent guest tenants.
pub const MAX_GUESTS: usize = 16;

/// Size of the software command ring per guest (in DWORDs).
pub const RING_SIZE_DW: usize = 4096;

// ── AegisHypervisor ──────────────────────────────────────────────────────────

pub struct AegisHypervisor {
    hal:         Arc<GpuHal>,
    bar_shadow:  Arc<BarShadow>,
    guests:      Arc<RwLock<HashMap<GuestId, Arc<Mutex<GuestContext>>>>>,
    scheduler:   Arc<Mutex<FairScheduler>>,
    mitigator:   Arc<SideChannelMitigator>,
    ptx_rewriter: Arc<PtxRewriter>,
    validator:   Arc<PacketValidator>,
    /// Monotonic epoch incremented on every context switch for replay-guard.
    epoch:       Arc<std::sync::atomic::AtomicU64>,
}

impl AegisHypervisor {
    /// Construct hypervisor and map BAR0 shadow region.
    pub async fn new(hal: GpuHal) -> Result<Self> {
        let hal = Arc::new(hal);

        let bar_shadow = Arc::new(
            BarShadow::map(Arc::clone(&hal))
                .context("Failed to map BAR0 shadow")?,
        );

        let mitigator = Arc::new(
            SideChannelMitigator::new(Arc::clone(&hal))
                .context("SideChannelMitigator init failed")?,
        );

        let ptx_rewriter = Arc::new(PtxRewriter::new());
        let validator    = Arc::new(PacketValidator::new(Arc::clone(&hal)));
        let scheduler    = Arc::new(Mutex::new(FairScheduler::new(MAX_GUESTS)));

        Ok(Self {
            hal,
            bar_shadow,
            guests:   Arc::new(RwLock::new(HashMap::with_capacity(MAX_GUESTS))),
            scheduler,
            mitigator,
            ptx_rewriter,
            validator,
            epoch: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        })
    }

    /// Main event loop — processes guest connects and command packets.
    pub async fn run(self, mut ipc: IpcServer) -> Result<()> {
        let hv = Arc::new(self);

        loop {
            tokio::select! {
                // ── New guest connection ──────────────────────────────────
                Ok(conn) = ipc.accept() => {
                    let hv2 = Arc::clone(&hv);
                    tokio::spawn(async move {
                        if let Err(e) = hv2.handle_guest_session(conn).await {
                            warn!("Guest session error: {e:#}");
                        }
                    });
                }

                // ── Scheduler tick ───────────────────────────────────────
                _ = tokio::time::sleep(tokio::time::Duration::from_micros(250)) => {
                    let hv2 = Arc::clone(&hv);
                    tokio::spawn(async move {
                        if let Err(e) = hv2.scheduler_tick().await {
                            warn!("Scheduler tick error: {e:#}");
                        }
                    });
                }
            }
        }
    }

    // ── Internal ─────────────────────────────────────────────────────────────

    #[instrument(skip(self, conn), fields(guest_id))]
    async fn handle_guest_session(
        &self,
        mut conn: crate::ipc::GuestConnection,
    ) -> Result<()> {
        // Allocate isolated memory arena for this guest
        let arena = IsolatedArena::allocate(
            Arc::clone(&self.hal),
            conn.requested_vram_bytes(),
        )
        .context("IsolatedArena allocation failed")?;

        let id = GuestId::new();
        tracing::Span::current().record("guest_id", &id.0);
        info!("Guest {id:?} connected, arena = {:?}", arena.range());

        let ctx = Arc::new(Mutex::new(
            GuestContext::new(id, arena, conn.priority())
        ));

        {
            let mut guests = self.guests.write().await;
            if guests.len() >= MAX_GUESTS {
                bail!("Guest limit ({MAX_GUESTS}) reached");
            }
            guests.insert(id, Arc::clone(&ctx));
        }

        self.scheduler.lock().await.register(id, conn.priority());

        // Per-guest packet processing loop
        loop {
            match conn.recv_packet().await {
                Ok(raw) => {
                    self.process_packet(id, raw).await?;
                }
                Err(crate::ipc::IpcError::Disconnected) => {
                    info!("Guest {id:?} disconnected");
                    break;
                }
                Err(e) => return Err(e.into()),
            }
        }

        self.evict_guest(id).await;
        Ok(())
    }

    /// Validate, rewrite (if PTX), and forward a single command packet.
    #[inline(always)]
    async fn process_packet(
        &self,
        guest_id: GuestId,
        raw: Vec<u8>,
    ) -> Result<()> {
        // Step 1: Structural validation — rejects mal-formed or oversized pkts
        let packet = self.validator.validate(&raw, guest_id)
            .context("Packet validation failed")?;

        // Step 2: If the packet contains a PTX kernel launch, rewrite it to
        //         insert yield points and bounds checks before dispatch.
        let packet = if packet.contains_ptx_launch() {
            self.ptx_rewriter.rewrite_launch(packet, guest_id)?
        } else {
            packet
        };

        // Step 3: Shadow-write to the per-guest command ring
        self.bar_shadow.enqueue(guest_id, &packet)?;

        debug!("Guest {guest_id:?} — packet dispatched (method=0x{:04X})",
               packet.method());
        Ok(())
    }

    /// Cooperative preemption: switch to the next guest chosen by the scheduler.
    async fn scheduler_tick(&self) -> Result<()> {
        let decision = {
            let mut sched = self.scheduler.lock().await;
            sched.next_context()
        };

        match decision {
            ScheduleDecision::Switch { from, to } => {
                self.context_switch(from, to).await?;
            }
            ScheduleDecision::Continue => {}
            ScheduleDecision::Idle => {}
        }
        Ok(())
    }

    /// Full context-switch: save → mitigate → restore.
    #[instrument(skip(self))]
    async fn context_switch(
        &self,
        from: Option<GuestId>,
        to: GuestId,
    ) -> Result<()> {
        let epoch = self.epoch
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        debug!("Context switch epoch={epoch}: {:?} → {:?}", from, to);

        // 1. Save outgoing context state into its GuestContext
        if let Some(prev_id) = from {
            if let Some(ctx) = self.guests.read().await.get(&prev_id).cloned() {
                ctx.lock().await.save(&self.hal)?;
            }
        }

        // 2. ══ SECURITY CRITICAL ══
        //    Flush all shared GPU state (L2 cache, shared memory, register file
        //    shadows) before loading the next guest's context.
        self.mitigator.full_flush(epoch).await
            .context("Side-channel flush failed")?;

        // 3. Restore incoming context
        if let Some(ctx) = self.guests.read().await.get(&to).cloned() {
            ctx.lock().await.restore(&self.hal)?;
        }

        self.bar_shadow.switch_active_channel(to)?;

        info!("Context switch complete → guest {:?}", to);
        Ok(())
    }

    async fn evict_guest(&self, id: GuestId) {
        self.guests.write().await.remove(&id);
        let _ = self.scheduler.lock().await.deregister(id);
        info!("Guest {id:?} evicted");
    }
}
