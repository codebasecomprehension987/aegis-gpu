// aegis-gpu/src/ptx_rewriter/mod.rs
// ═══════════════════════════════════════════════════════════════════════════
//  Sub-Kernel Partitioning via On-the-Fly PTX Rewriting
//
//  When a guest submits a kernel launch, this module intercepts the PTX
//  source (or SASS binary via the PTX-to-SASS JIT path) and:
//
//  1. Parses the instruction stream with a hand-written nom combinator.
//  2. Inserts cooperative yield points (`nanosleep` / `membar.cta`) every
//     N instructions to allow software preemption.
//  3. Adds bounds-check predicates before every LD/ST global instruction
//     to ensure the access falls within the guest's arena.
//  4. Strips any `ld.global` accessing addresses outside the arena, replacing
//     them with a trap that signals the hypervisor.
//  5. Re-assembles the modified PTX for NVPTX compilation.
// ═══════════════════════════════════════════════════════════════════════════

pub mod parser;
pub mod inserter;
pub mod assembler;

use anyhow::{Context, Result};
use tracing::{debug, trace};

use crate::hypervisor::{GuestId, packet_validator::ValidatedPacket};
use parser::{PtxKernel, PtxInstruction};
use inserter::YieldInserter;
use assembler::PtxAssembler;

/// Instruction budget between consecutive yield points.
/// Lower = finer preemption granularity but higher overhead.
pub const YIELD_INTERVAL: usize = 256;

/// Max PTX source size accepted from a guest (4 MiB).
pub const MAX_PTX_BYTES: usize = 4 * 1024 * 1024;

// ── PtxRewriter ───────────────────────────────────────────────────────────────

pub struct PtxRewriter {
    /// Re-usable scratch buffer for assembler output.
    scratch: std::sync::Mutex<Vec<u8>>,
}

impl PtxRewriter {
    pub fn new() -> Self {
        Self {
            scratch: std::sync::Mutex::new(Vec::with_capacity(64 * 1024)),
        }
    }

    /// Rewrite the PTX embedded in a kernel-launch packet.
    ///
    /// The returned packet is a modified copy; the original is consumed.
    pub fn rewrite_launch(
        &self,
        packet: ValidatedPacket,
        guest_id: GuestId,
    ) -> Result<ValidatedPacket> {
        let ptx_bytes = extract_ptx_from_packet(&packet)
            .context("Failed to extract PTX from launch packet")?;

        if ptx_bytes.len() > MAX_PTX_BYTES {
            anyhow::bail!(
                "PTX too large: {} bytes (max {})",
                ptx_bytes.len(), MAX_PTX_BYTES
            );
        }

        let ptx_src = std::str::from_utf8(ptx_bytes)
            .context("PTX payload is not valid UTF-8")?;

        debug!("Rewriting PTX for guest {:?} ({} bytes)", guest_id, ptx_src.len());

        // ── Parse ─────────────────────────────────────────────────────────────
        let mut kernel = parser::parse_ptx(ptx_src)
            .context("PTX parse error")?;

        let original_instr_count: usize = kernel.functions.iter()
            .map(|f| f.body.len())
            .sum();

        // ── Insert yield points ───────────────────────────────────────────────
        let mut inserter = YieldInserter::new(YIELD_INTERVAL);
        for func in &mut kernel.functions {
            inserter.insert_yields(&mut func.body);
        }

        // ── Insert memory bounds checks ───────────────────────────────────────
        let arena = self.get_arena_bounds(guest_id);
        for func in &mut kernel.functions {
            self.insert_bounds_checks(&mut func.body, arena);
        }

        let rewritten_count: usize = kernel.functions.iter()
            .map(|f| f.body.len())
            .sum();

        debug!(
            "PTX rewrite complete: {original_instr_count} → {rewritten_count} instructions"
        );

        // ── Re-assemble ───────────────────────────────────────────────────────
        let mut scratch = self.scratch.lock().unwrap();
        scratch.clear();
        PtxAssembler::emit(&kernel, &mut *scratch)?;

        let new_ptx = scratch.clone();
        drop(scratch);

        // Re-embed modified PTX into a new packet
        let new_packet = embed_ptx_into_packet(packet, &new_ptx)?;
        Ok(new_packet)
    }

    // ── Private ───────────────────────────────────────────────────────────────

    fn get_arena_bounds(&self, _guest_id: GuestId) -> (u64, u64) {
        // In a real implementation this would query the memory subsystem.
        // Returns (base, top) of the guest's VRAM arena.
        (0x0001_0000_0000, 0x0002_0000_0000) // placeholder 4 GiB window
    }

    /// For every `ld.global` / `st.global` instruction, prepend:
    ///
    /// ```ptx
    ///   setp.lo.u64 %p_chk, %addr, ARENA_TOP   // addr < top?
    ///   setp.ge.u64 %p_chk2, %addr, ARENA_BASE // addr >= base?
    ///   and.pred    %p_ok, %p_chk, %p_chk2
    ///   @!%p_ok trap;                           // out-of-bounds → trap
    /// ```
    fn insert_bounds_checks(
        &self,
        body: &mut Vec<PtxInstruction>,
        (base, top): (u64, u64),
    ) {
        let mut insertions: Vec<(usize, Vec<PtxInstruction>)> = Vec::new();

        for (i, instr) in body.iter().enumerate() {
            if instr.is_global_mem_access() {
                if let Some(addr_reg) = instr.address_register() {
                    let guard = vec![
                        PtxInstruction::set_pred_lo_u64(
                            "%_p_chk_hi", addr_reg, top),
                        PtxInstruction::set_pred_ge_u64(
                            "%_p_chk_lo", addr_reg, base),
                        PtxInstruction::and_pred(
                            "%_p_ok", "%_p_chk_hi", "%_p_chk_lo"),
                        PtxInstruction::conditional_trap("%_p_ok"),
                    ];
                    insertions.push((i, guard));
                }
            }
        }

        // Insert in reverse order to preserve indices
        for (idx, guards) in insertions.into_iter().rev() {
            for (j, g) in guards.into_iter().enumerate() {
                body.insert(idx + j, g);
            }
        }
        trace!("Bounds checks inserted into {} ld/st sites", insertions.len());
    }
}

// ── Packet helpers (stubs — real impl would parse NVGPU packet format) ────────

fn extract_ptx_from_packet(packet: &ValidatedPacket) -> Result<&[u8]> {
    // In production: locate the PTX payload DWORDs within the command packet
    // using the NVC6C0_SET_PROGRAM_OBJECT method offset.
    let dwords = packet.dwords();
    if dwords.len() < 4 {
        anyhow::bail!("Launch packet too short for PTX extraction");
    }
    // Byte offset 16 (DW[4]) onwards is the embedded PTX region.
    let byte_offset = 4 * 4;
    let raw_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            dwords.as_ptr().cast::<u8>().add(byte_offset),
            (dwords.len() - 4) * 4,
        )
    };
    Ok(raw_bytes)
}

fn embed_ptx_into_packet(mut packet: ValidatedPacket, ptx: &[u8]) -> Result<ValidatedPacket> {
    // In production: rebuild the packet with the new PTX bytes.
    // For this blueprint we return the packet unchanged.
    let _ = ptx;
    Ok(packet)
}
