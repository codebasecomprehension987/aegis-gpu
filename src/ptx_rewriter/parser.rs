// aegis-gpu/src/ptx_rewriter/parser.rs
// PTX 8.x grammar parser built on nom combinators.
// Handles enough of the PTX ISA to support the rewriter's requirements.

use nom::{
    IResult,
    branch::alt,
    bytes::complete::{tag, take_while1, take_until},
    character::complete::{alpha1, alphanumeric1, char, multispace0, multispace1, digit1},
    combinator::{map, opt, recognize},
    multi::{many0, many1, separated_list0},
    sequence::{delimited, preceded, terminated, tuple},
};

// ── AST ───────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct PtxKernel {
    pub version:   (u32, u32),
    pub target:    String,
    pub functions: Vec<PtxFunction>,
}

#[derive(Debug, Clone)]
pub struct PtxFunction {
    pub name:   String,
    pub params: Vec<PtxParam>,
    pub body:   Vec<PtxInstruction>,
}

#[derive(Debug, Clone)]
pub struct PtxParam {
    pub name:  String,
    pub space: String,
    pub ty:    String,
}

/// A single PTX instruction (simplified IR).
#[derive(Debug, Clone)]
pub struct PtxInstruction {
    pub predicate: Option<String>,
    pub opcode:    String,
    pub modifiers: Vec<String>,
    pub operands:  Vec<String>,
    /// True if this was synthesised by the rewriter (not from guest source).
    pub synthetic: bool,
}

impl PtxInstruction {
    // ── Query helpers ─────────────────────────────────────────────────────────

    pub fn is_global_mem_access(&self) -> bool {
        matches!(self.opcode.as_str(), "ld" | "st" | "atom" | "red")
            && self.modifiers.iter().any(|m| m == "global")
    }

    pub fn address_register(&self) -> Option<&str> {
        // Address operand for ld/st is typically operands[1] enclosed in [...]
        for op in &self.operands {
            if op.starts_with('[') && op.ends_with(']') {
                return Some(op.trim_start_matches('[').trim_end_matches(']'));
            }
        }
        None
    }

    pub fn is_branch(&self) -> bool {
        matches!(self.opcode.as_str(), "bra" | "brx" | "call" | "ret")
    }

    // ── Synthetic instruction constructors ────────────────────────────────────

    pub fn nanosleep(cycles: u32) -> Self {
        Self {
            predicate: None,
            opcode: "nanosleep".into(),
            modifiers: vec![],
            operands: vec![cycles.to_string()],
            synthetic: true,
        }
    }

    pub fn membar_cta() -> Self {
        Self {
            predicate: None,
            opcode: "membar".into(),
            modifiers: vec!["cta".into()],
            operands: vec![],
            synthetic: true,
        }
    }

    pub fn set_pred_lo_u64(pred: &str, reg: &str, imm: u64) -> Self {
        Self {
            predicate: None,
            opcode: "setp.lo.u64".into(),
            modifiers: vec![],
            operands: vec![pred.into(), reg.into(), format!("{imm:#X}")],
            synthetic: true,
        }
    }

    pub fn set_pred_ge_u64(pred: &str, reg: &str, imm: u64) -> Self {
        Self {
            predicate: None,
            opcode: "setp.ge.u64".into(),
            modifiers: vec![],
            operands: vec![pred.into(), reg.into(), format!("{imm:#X}")],
            synthetic: true,
        }
    }

    pub fn and_pred(out: &str, a: &str, b: &str) -> Self {
        Self {
            predicate: None,
            opcode: "and.pred".into(),
            modifiers: vec![],
            operands: vec![out.into(), a.into(), b.into()],
            synthetic: true,
        }
    }

    /// `@!%pred trap` — fires when predicate is FALSE (i.e., out-of-bounds).
    pub fn conditional_trap(ok_pred: &str) -> Self {
        Self {
            predicate: Some(format!("!{ok_pred}")),
            opcode: "trap".into(),
            modifiers: vec![],
            operands: vec![],
            synthetic: true,
        }
    }
}

// ── Parser ────────────────────────────────────────────────────────────────────

pub fn parse_ptx(input: &str) -> anyhow::Result<PtxKernel> {
    match ptx_file(input) {
        Ok((_, kernel)) => Ok(kernel),
        Err(e) => anyhow::bail!("PTX parse error: {e}"),
    }
}

fn ptx_file(i: &str) -> IResult<&str, PtxKernel> {
    let (i, _) = multispace0(i)?;
    let (i, version) = ptx_version(i)?;
    let (i, _) = multispace0(i)?;
    let (i, target) = ptx_target(i)?;
    let (i, _) = multispace0(i)?;
    let (i, _) = opt(ptx_address_size)(i)?;
    let (i, _) = multispace0(i)?;
    let (i, functions) = many0(preceded(multispace0, ptx_function))(i)?;

    Ok((i, PtxKernel { version, target: target.to_string(), functions }))
}

fn ptx_version(i: &str) -> IResult<&str, (u32, u32)> {
    let (i, _) = tag(".version")(i)?;
    let (i, _) = multispace1(i)?;
    let (i, major) = map(digit1, |s: &str| s.parse::<u32>().unwrap_or(0))(i)?;
    let (i, _) = char('.')(i)?;
    let (i, minor) = map(digit1, |s: &str| s.parse::<u32>().unwrap_or(0))(i)?;
    Ok((i, (major, minor)))
}

fn ptx_target(i: &str) -> IResult<&str, &str> {
    let (i, _) = tag(".target")(i)?;
    let (i, _) = multispace1(i)?;
    let (i, t) = take_while1(|c: char| c.is_alphanumeric() || c == '_')(i)?;
    Ok((i, t))
}

fn ptx_address_size(i: &str) -> IResult<&str, &str> {
    let (i, _) = tag(".address_size")(i)?;
    let (i, _) = multispace1(i)?;
    let (i, v) = digit1(i)?;
    Ok((i, v))
}

fn ptx_function(i: &str) -> IResult<&str, PtxFunction> {
    // .entry or .func
    let (i, _) = alt((tag(".entry"), tag(".func"), tag(".visible .entry")))(i)?;
    let (i, _) = multispace1(i)?;
    let (i, name) = identifier(i)?;
    let (i, _) = multispace0(i)?;
    let (i, params) = opt(delimited(
        char('('),
        separated_list0(
            preceded(multispace0, char(',')),
            preceded(multispace0, ptx_param),
        ),
        preceded(multispace0, char(')')),
    ))(i)?;
    let (i, _) = multispace0(i)?;
    let (i, body) = ptx_block(i)?;

    Ok((i, PtxFunction {
        name: name.to_string(),
        params: params.unwrap_or_default(),
        body,
    }))
}

fn ptx_param(i: &str) -> IResult<&str, PtxParam> {
    let (i, _) = opt(preceded(tag(".param"), multispace1))(i)?;
    let (i, space) = opt(preceded(char('.'), alpha1))(i)?;
    let (i, _) = multispace0(i)?;
    let (i, ty) = take_while1(|c: char| c.is_alphanumeric() || c == '.')(i)?;
    let (i, _) = multispace1(i)?;
    let (i, name) = identifier(i)?;
    Ok((i, PtxParam {
        name: name.to_string(),
        space: space.unwrap_or("").to_string(),
        ty: ty.to_string(),
    }))
}

fn ptx_block(i: &str) -> IResult<&str, Vec<PtxInstruction>> {
    delimited(
        preceded(multispace0, char('{')),
        many0(preceded(multispace0, ptx_instruction)),
        preceded(multispace0, char('}')),
    )(i)
}

fn ptx_instruction(i: &str) -> IResult<&str, PtxInstruction> {
    // Skip labels (identifier followed by ':')
    let (i, _) = opt(terminated(identifier, preceded(multispace0, char(':'))))(i)?;
    let (i, _) = multispace0(i)?;

    // Optional predicate: @%pred or @!%pred
    let (i, predicate) = opt(map(
        preceded(
            char('@'),
            take_while1(|c: char| c.is_alphanumeric() || c == '_' || c == '%' || c == '!'),
        ),
        String::from,
    ))(i)?;
    let (i, _) = multispace0(i)?;

    // Opcode (with optional dot-separated modifiers)
    let (i, opcode_str) = take_while1(|c: char| c.is_alphanumeric() || c == '.' || c == '_')(i)?;

    let parts: Vec<&str> = opcode_str.splitn(2, '.').collect();
    let opcode     = parts[0].to_string();
    let modifiers  = if parts.len() > 1 {
        parts[1].split('.').map(String::from).collect()
    } else {
        vec![]
    };

    let (i, _) = multispace0(i)?;

    // Operands separated by commas, terminated by ';'
    let (i, operands) = many0(map(
        terminated(
            take_while1(|c: char| c != ',' && c != ';' && c != '\n'),
            opt(preceded(multispace0, char(','))),
        ),
        |s: &str| s.trim().to_string(),
    ))(i)?;

    let (i, _) = opt(preceded(multispace0, char(';')))(i)?;

    Ok((i, PtxInstruction {
        predicate,
        opcode,
        modifiers,
        operands,
        synthetic: false,
    }))
}

fn identifier(i: &str) -> IResult<&str, &str> {
    recognize(tuple((
        alt((alpha1, tag("_"), tag("%"))),
        many0(alt((alphanumeric1, tag("_"), tag("$")))),
    )))(i)
}
