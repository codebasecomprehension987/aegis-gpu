// aegis-gpu/src/scheduler/mod.rs
// ═══════════════════════════════════════════════════════════════════════════
//  Weighted Deficit Round-Robin (WDRR) Scheduler
//
//  Provides strict computational fairness across N guest tenants.  Each
//  guest is assigned a `weight` (1–15) which determines its quantum share.
//  The scheduler tracks a per-guest "deficit counter" that accumulates
//  unused quanta, preventing starvation of low-priority tenants.
//
//  Reference: Shreedhar & Varghese, "Efficient Fair Queuing Using Deficit
//  Round-Robin", IEEE/ACM ToN 1996.
// ═══════════════════════════════════════════════════════════════════════════

use std::collections::VecDeque;
use std::time::{Duration, Instant};
use tracing::{debug, trace};
use crate::hypervisor::GuestId;

// ── Constants ─────────────────────────────────────────────────────────────────

/// Base quantum in microseconds.  Each unit of weight = QUANTUM_BASE µs.
const QUANTUM_BASE_US: u64 = 50;

/// Maximum accumulated deficit (in µs) to prevent burst after long idle.
const MAX_DEFICIT_US: u64 = 5_000;

// ── Types ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScheduleDecision {
    /// Switch from `from` (if any) to `to`.
    Switch { from: Option<GuestId>, to: GuestId },
    /// Current context is within quantum — no switch needed.
    Continue,
    /// No runnable guests.
    Idle,
}

#[derive(Debug, Clone)]
struct QueueEntry {
    id:           GuestId,
    weight:       u8,
    deficit_us:   u64,
    total_gpu_us: u64,
    last_run:     Option<Instant>,
}

// ── FairScheduler ─────────────────────────────────────────────────────────────

pub struct FairScheduler {
    queue:        VecDeque<QueueEntry>,
    active:       Option<GuestId>,
    active_start: Option<Instant>,
    max_guests:   usize,
}

impl FairScheduler {
    pub fn new(max_guests: usize) -> Self {
        Self {
            queue:        VecDeque::with_capacity(max_guests),
            active:       None,
            active_start: None,
            max_guests,
        }
    }

    /// Register a new guest with `priority` (weight = 1..=15).
    pub fn register(&mut self, id: GuestId, priority: u8) {
        if self.queue.len() >= self.max_guests {
            tracing::warn!("Scheduler at capacity, refusing registration of {:?}", id);
            return;
        }
        let weight = priority.clamp(1, 15);
        self.queue.push_back(QueueEntry {
            id,
            weight,
            deficit_us: (weight as u64) * QUANTUM_BASE_US, // initial burst
            total_gpu_us: 0,
            last_run: None,
        });
        debug!("Registered guest {:?} weight={weight}", id);
    }

    /// Remove a guest from the scheduler.
    pub fn deregister(&mut self, id: GuestId) -> bool {
        if let Some(pos) = self.queue.iter().position(|e| e.id == id) {
            self.queue.remove(pos);
            if self.active == Some(id) {
                self.active = None;
                self.active_start = None;
            }
            debug!("Deregistered guest {:?}", id);
            true
        } else {
            false
        }
    }

    /// Determine the next scheduling action.
    ///
    /// Called every 250 µs by the hypervisor timer.
    pub fn next_context(&mut self) -> ScheduleDecision {
        if self.queue.is_empty() {
            return ScheduleDecision::Idle;
        }

        let now = Instant::now();

        // Account for time used by current active context
        if let (Some(active_id), Some(start)) = (self.active, self.active_start) {
            let elapsed_us = start.elapsed().as_micros() as u64;
            if let Some(entry) = self.queue.iter_mut().find(|e| e.id == active_id) {
                entry.total_gpu_us += elapsed_us;
                entry.deficit_us = entry.deficit_us.saturating_sub(elapsed_us);
                trace!(
                    "Guest {:?} used {elapsed_us}µs, deficit={}µs",
                    active_id, entry.deficit_us
                );

                // If deficit remains, continue running
                if entry.deficit_us > 0 {
                    self.active_start = Some(now);
                    return ScheduleDecision::Continue;
                }
            }
        }

        // Find the next guest with positive or replenishable deficit (WDRR)
        let next = self.pick_next();

        match next {
            None => ScheduleDecision::Idle,
            Some(next_id) if Some(next_id) == self.active => {
                self.active_start = Some(now);
                ScheduleDecision::Continue
            }
            Some(next_id) => {
                let from = self.active;
                self.active = Some(next_id);
                self.active_start = Some(now);
                debug!("Schedule: {:?} → {:?}", from, next_id);
                ScheduleDecision::Switch { from, to: next_id }
            }
        }
    }

    // ── Internal WDRR logic ───────────────────────────────────────────────────

    fn pick_next(&mut self) -> Option<GuestId> {
        // Pass 1: any guest with existing positive deficit
        if let Some(entry) = self.queue.iter_mut().find(|e| e.deficit_us > 0) {
            return Some(entry.id);
        }

        // Pass 2: replenish all deficits by weight and pick the largest
        for entry in &mut self.queue {
            let replenish = (entry.weight as u64) * QUANTUM_BASE_US;
            entry.deficit_us = (entry.deficit_us + replenish).min(MAX_DEFICIT_US);
        }

        // Sort by (deficit DESC, total_gpu_us ASC) for fairness
        let mut candidates: Vec<&QueueEntry> = self.queue.iter().collect();
        candidates.sort_by(|a, b| {
            b.deficit_us.cmp(&a.deficit_us)
                .then(a.total_gpu_us.cmp(&b.total_gpu_us))
        });

        candidates.first().map(|e| e.id)
    }
}
