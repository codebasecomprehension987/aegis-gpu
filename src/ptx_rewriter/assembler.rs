// aegis-gpu/src/ptx_rewriter/assembler.rs
// Re-serialises a PtxKernel AST back to PTX source text.

use std::io::Write;
use anyhow::Result;
use super::parser::{PtxKernel, PtxInstruction};

pub struct PtxAssembler;

impl PtxAssembler {
    pub fn emit(kernel: &PtxKernel, out: &mut Vec<u8>) -> Result<()> {
        write!(out, ".version {}.{}\n", kernel.version.0, kernel.version.1)?;
        write!(out, ".target {}\n", kernel.target)?;
        write!(out, ".address_size 64\n\n")?;

        for func in &kernel.functions {
            write!(out, ".visible .entry {}(\n", func.name)?;
            for (i, p) in func.params.iter().enumerate() {
                let comma = if i + 1 < func.params.len() { "," } else { "" };
                write!(out, "    .param .{} {}{}\n", p.ty, p.name, comma)?;
            }
            write!(out, ")\n{{\n")?;

            for instr in &func.body {
                Self::emit_instr(instr, out)?;
            }

            write!(out, "}}\n\n")?;
        }
        Ok(())
    }

    fn emit_instr(instr: &PtxInstruction, out: &mut Vec<u8>) -> Result<()> {
        write!(out, "    ")?;
        if let Some(pred) = &instr.predicate {
            write!(out, "@{pred} ")?;
        }
        write!(out, "{}", instr.opcode)?;
        for m in &instr.modifiers {
            write!(out, ".{m}")?;
        }
        if !instr.operands.is_empty() {
            write!(out, " {}", instr.operands.join(", "))?;
        }
        write!(out, ";\n")?;
        Ok(())
    }
}
