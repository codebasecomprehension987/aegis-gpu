// aegis-gpu/src/ptx_rewriter/inserter.rs
// Inserts nanosleep/membar yield points every N instructions.

use super::parser::PtxInstruction;

pub struct YieldInserter {
    interval: usize,
}

impl YieldInserter {
    pub fn new(interval: usize) -> Self {
        Self { interval }
    }

    /// Walk `body` and insert a yield sequence after every `self.interval`
    /// non-synthetic, non-branch instructions.
    pub fn insert_yields(&mut self, body: &mut Vec<PtxInstruction>) {
        let mut count = 0usize;
        let mut insertions: Vec<usize> = Vec::new();

        for (i, instr) in body.iter().enumerate() {
            if instr.synthetic { continue; }
            count += 1;
            if count % self.interval == 0 {
                insertions.push(i + 1);
            }
        }

        // Insert in reverse to preserve earlier indices
        for &idx in insertions.iter().rev() {
            body.insert(idx, PtxInstruction::nanosleep(16));
            body.insert(idx, PtxInstruction::membar_cta());
        }
    }
}
