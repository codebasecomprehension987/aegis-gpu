# Aegis-GPU

A Rust-based microkernel for secure, multi-tenant GPU execution.

Aegis-GPU treats the GPU as a first-class virtualized resource. Instead of
allowing guest libraries direct access to hardware, it interposes on every
command packet via PCIe BAR0 shadowing, rewrites guest PTX kernels on the
fly to insert preemption points and memory bounds checks, and performs
cryptographic side-channel mitigation between every context switch.

---

## Architecture

```
Guest user-space (libcuda shim)
        │  Unix socket (IPC)
        ▼
┌─────────────────────────────────┐
│        AegisHypervisor          │
│  ┌──────────┐  ┌─────────────┐  │
│  │  Packet  │  │ PTX Rewriter│  │
│  │Validator │  │ (nom parser)│  │
│  └────┬─────┘  └──────┬──────┘  │
│       │               │         │
│  ┌────▼───────────────▼──────┐  │
│  │       BAR0 Shadow Ring    │  │
│  └────────────┬──────────────┘  │
│               │  MMIO replay    │
│  ┌────────────▼──────────────┐  │
│  │   SideChannelMitigator    │  │  ← between every context switch
│  │  L2 flush │ scrub kernel  │  │
│  └───────────────────────────┘  │
│  ┌───────────────────────────┐  │
│  │     WDRR Fair Scheduler   │  │
│  └───────────────────────────┘  │
└────────────────┬────────────────┘
                 │  validated MMIO writes
                 ▼
           Real GPU Hardware
```

## Features

| Subsystem | Technique |
|-----------|-----------|
| Command interception | PCIe BAR0 shadow rings with per-guest atomic queues |
| Packet validation | Method whitelist + BLAKE3 checksum + 64-bit VA bounds check |
| Preemption | PTX rewriter inserts `nanosleep`+`membar.cta` every 256 instructions |
| Memory safety | `IsolatedArena` with zeroed-on-alloc and zeroed-on-free |
| Fairness | Weighted Deficit Round-Robin scheduler (250 µs tick) |
| Side channels | L2 LTC hardware flush + CUDA scrub kernel + L1 invalidate |
| Formal proofs | Kani harnesses for arena overlap, OOB rejection, scheduler liveness |

---

## Requirements

| Tool | Version | Notes |
|------|---------|-------|
| Rust nightly | see `rust-toolchain.toml` | Required for Kani |
| CUDA Toolkit | 12.x+ | Optional — simulation mode if absent |
| Linux | 5.15+ | `/dev/mem` or UIO for BAR0 mapping |
| NVIDIA GPU | Ampere (sm_80) or newer | RTX 30xx / A100 / H100 |

---

## Build

```bash
# Standard build
cargo build --release

# With formal verification (requires cargo-kani)
cargo install kani-verifier
cargo kani --features verify

# Run tests (no GPU required)
cargo test

# Benchmarks
cargo bench
```

---

## Running

```bash
# Must be run as root or with CAP_SYS_RAWIO for BAR0 access
sudo ./target/release/aegis-hypervisor

# Guests connect via /run/aegis-gpu.sock
# Set log level with RUST_LOG=debug
RUST_LOG=aegis_gpu=debug sudo ./target/release/aegis-hypervisor
```

---

## Security Model

Aegis-GPU assumes the GPU hardware itself is trusted. The threat model covers:

- **Malicious guest code** attempting out-of-bounds VRAM access
- **Timing side-channels** via shared L2 cache and shared memory banks
- **Replay attacks** via packet sequence-number monotonicity enforcement
- **Privilege escalation** via forbidden MMIO methods (whitelist-deny approach)

See [`docs/SECURITY.md`](docs/SECURITY.md) for the full threat model and
formal verification coverage matrix.

---

## License

MIT — see [LICENSE](LICENSE).
