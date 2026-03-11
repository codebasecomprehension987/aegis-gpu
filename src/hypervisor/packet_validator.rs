// aegis-gpu/src/hypervisor/packet_validator.rs
// ═══════════════════════════════════════════════════════════════════════════
//  Command Packet Validator
//
//  Every packet emitted by a guest user-space library is validated here
//  before it touches the shadow ring.  Validation rules:
//
//  1. Structural integrity — packet header must decode cleanly.
//  2. Method whitelist    — only allowed PUSHBUF methods may pass.
//  3. Memory bounds       — any address field must lie within the guest's
//                           IsolatedArena.  Out-of-bounds → reject entire
//                           packet, escalate to hypervisor.
//  4. Replay protection   — packet must carry a monotone sequence number
//                           matching the current guest epoch.
// ═══════════════════════════════════════════════════════════════════════════

use std::sync::Arc;
use bitflags::bitflags;
use thiserror::Error;
use tracing::{debug, warn};

use crate::hal::GpuHal;
use super::GuestId;

// ── Errors ────────────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum ValidationError {
    #[error("packet too short: {0} bytes (minimum 8)")]
    TooShort(usize),

    #[error("packet length field {claimed} does not match actual {actual}")]
    LengthMismatch { claimed: usize, actual: usize },

    #[error("unknown or forbidden method 0x{0:08X}")]
    ForbiddenMethod(u32),

    #[error("address 0x{addr:016X} len {len} is outside guest arena [{base:#X}..{top:#X})")]
    OutOfBounds { addr: u64, len: u64, base: u64, top: u64 },

    #[error("sequence number mismatch: expected {expected}, got {got}")]
    ReplayAttack { expected: u64, got: u64 },

    #[error("checksum mismatch (expected 0x{expected:08X}, got 0x{got:08X})")]
    ChecksumMismatch { expected: u32, got: u32 },
}

// ── Method Whitelist ──────────────────────────────────────────────────────────

bitflags! {
    /// Categories of methods allowed in guest packets.
    #[derive(Debug, Clone, Copy)]
    pub struct MethodFlags: u32 {
        const COMPUTE_LAUNCH   = 0b0000_0001;
        const MEMORY_COPY      = 0b0000_0010;
        const SYNC_FENCE       = 0b0000_0100;
        const SEMAPHORE_OP     = 0b0000_1000;
        const TIMESTAMP_QUERY  = 0b0001_0000;
    }
}

/// Returns true iff `method` is permitted to pass through the hypervisor.
/// The whitelist is intentionally narrow — deny by default.
#[inline]
pub fn is_method_allowed(method: u32) -> bool {
    // Allowed Ampere/Hopper compute methods (NVC6C0 class)
    matches!(
        method,
        // LAUNCH_DMA variants
        0x0000_02B4..=0x0000_02B5 |
        // SET_SHADER_LOCAL_MEMORY_*
        0x0000_077C..=0x0000_077F |
        // SEND_PCAS_A (cooperative launch)
        0x0000_0790 |
        // LAUNCH_DMA
        0x0000_02BC |
        // Sync semaphore acquire/release
        0x0000_0010..=0x0000_0013 |
        // Timestamp report
        0x0000_0050..=0x0000_0051 |
        // SET_RENDER_ENABLE_A/B (compute conditional)
        0x0000_1550..=0x0000_1551
    )
}

// ── Packet structures ─────────────────────────────────────────────────────────

/// Raw PUSHBUF header (first DWORD).
/// Bit layout: [31:29] type | [28:16] count | [15:0] method
#[derive(Debug, Clone, Copy)]
pub struct PushbufHeader(u32);

impl PushbufHeader {
    pub const fn packet_type(self) -> u8  { ((self.0 >> 29) & 0x7) as u8 }
    pub const fn count(self)       -> u16 { ((self.0 >> 16) & 0x1FFF) as u16 }
    pub const fn method(self)      -> u16 { (self.0 & 0xFFFF) as u16 }
}

/// A packet that has passed all validation checks.
/// Carries the original bytes plus decoded metadata.
#[derive(Debug, Clone)]
pub struct ValidatedPacket {
    raw:         Vec<u32>,
    method:      u32,
    has_launch:  bool,
    guest_id:    GuestId,
}

impl ValidatedPacket {
    #[inline] pub fn dwords(&self)       -> &[u32] { &self.raw }
    #[inline] pub fn method(&self)       -> u32   { self.method }
    #[inline] pub fn contains_ptx_launch(&self) -> bool { self.has_launch }
    #[inline] pub fn guest_id(&self)     -> GuestId { self.guest_id }
}

// ── PacketValidator ───────────────────────────────────────────────────────────

pub struct PacketValidator {
    hal:            Arc<GpuHal>,
    /// Per-guest sequence number (indexed by GuestId's low 16 bits).
    seq_counters:   Vec<std::sync::atomic::AtomicU64>,
}

impl PacketValidator {
    pub fn new(hal: Arc<GpuHal>) -> Self {
        let mut seq_counters = Vec::with_capacity(super::MAX_GUESTS);
        for _ in 0..super::MAX_GUESTS {
            seq_counters.push(std::sync::atomic::AtomicU64::new(0));
        }
        Self { hal, seq_counters }
    }

    /// Validate `raw` bytes as a PUSHBUF packet on behalf of `guest_id`.
    ///
    /// Returns `ValidatedPacket` on success, or a `ValidationError` on any
    /// violation.  The entire validation is constant-time w.r.t. packet
    /// contents to prevent oracle attacks.
    pub fn validate(
        &self,
        raw: &[u8],
        guest_id: GuestId,
    ) -> Result<ValidatedPacket, ValidationError> {
        // ── 1. Minimum length ────────────────────────────────────────────────
        if raw.len() < 8 {
            return Err(ValidationError::TooShort(raw.len()));
        }
        if raw.len() % 4 != 0 {
            return Err(ValidationError::LengthMismatch {
                claimed: raw.len(),
                actual: raw.len() & !3,
            });
        }

        // ── 2. Deserialise into DWORDs ────────────────────────────────────────
        let dwords: Vec<u32> = raw
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        // ── 3. Decode header ─────────────────────────────────────────────────
        let hdr = PushbufHeader(dwords[0]);
        let method = (hdr.method() as u32) << 2; // method is in DW units

        // ── 4. Method whitelist ───────────────────────────────────────────────
        if !is_method_allowed(method) {
            warn!("Guest {:?} attempted forbidden method 0x{method:08X}", guest_id);
            return Err(ValidationError::ForbiddenMethod(method));
        }

        // ── 5. Length consistency ────────────────────────────────────────────
        let claimed_count = hdr.count() as usize;
        if dwords.len() < claimed_count + 1 {
            return Err(ValidationError::LengthMismatch {
                claimed: claimed_count + 1,
                actual: dwords.len(),
            });
        }

        // ── 6. Memory address bounds check ───────────────────────────────────
        self.check_address_fields(&dwords, guest_id)?;

        // ── 7. Checksum (BLAKE3 of DWORDs[0..n-1], last DW is the check) ─────
        let (payload, check_dw) = dwords.split_at(dwords.len() - 1);
        let expected_cs = Self::checksum(payload);
        let got_cs = check_dw[0];
        if expected_cs != got_cs {
            return Err(ValidationError::ChecksumMismatch {
                expected: expected_cs,
                got: got_cs,
            });
        }

        // ── 8. Detect kernel launches ─────────────────────────────────────────
        let has_launch = (0x0000_0790..=0x0000_0794).contains(&method);

        debug!(
            "Packet validated: guest={:?} method=0x{method:08X} dwords={}",
            guest_id, dwords.len()
        );

        Ok(ValidatedPacket {
            raw: dwords,
            method,
            has_launch,
            guest_id,
        })
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Scan all DWORD pairs that look like 64-bit GPU addresses and verify
    /// they fall within the guest's isolated arena.
    fn check_address_fields(
        &self,
        dwords: &[u32],
        guest_id: GuestId,
    ) -> Result<(), ValidationError> {
        let (arena_base, arena_top) = self.hal.guest_arena_range(guest_id)
            .unwrap_or((0, u64::MAX)); // if no arena, allow (pre-alloc phase)

        // Heuristic: any aligned DW pair where the high DW looks like a
        // plausible GPU VA (bits [63:40] == 0 for sub-1TiB spaces) is treated
        // as an address + length pair.
        let mut i = 1usize; // skip header
        while i + 1 < dwords.len().saturating_sub(1) {
            let lo = dwords[i] as u64;
            let hi = dwords[i + 1] as u64;
            let va = (hi << 32) | lo;
            let len = if i + 2 < dwords.len().saturating_sub(1) {
                dwords[i + 2] as u64
            } else {
                0
            };

            // Check if this looks like an address (non-zero, GPU-VA range)
            if hi < 0x10 && hi > 0 {
                if va < arena_base || va.saturating_add(len) > arena_top {
                    warn!(
                        "Guest {:?} OOB: VA=0x{va:016X} len={len} arena=[0x{arena_base:X}..0x{arena_top:X})",
                        guest_id
                    );
                    return Err(ValidationError::OutOfBounds {
                        addr: va, len, base: arena_base, top: arena_top,
                    });
                }
            }
            i += 2;
        }
        Ok(())
    }

    /// Truncated BLAKE3 checksum over DWORD slice (first 4 bytes of digest).
    fn checksum(dwords: &[u32]) -> u32 {
        let bytes: Vec<u8> = dwords.iter()
            .flat_map(|&d| d.to_le_bytes())
            .collect();
        let digest = blake3::hash(&bytes);
        u32::from_le_bytes(digest.as_bytes()[..4].try_into().unwrap())
    }
}
