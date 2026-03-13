// aegis-gpu/src/main.rs
// Aegis-GPU: Secure Multi-Tenant GPU Microkernel
// Entry point — initialises the hypervisor and blocks on the event loop.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::pedantic, clippy::nursery)]

mod hypervisor;
mod scheduler;
mod ptx_rewriter;
mod memory;
mod side_channel;
mod verification;
mod ipc;
mod hal;

use anyhow::Result;
use tracing::{info, error};
use tracing_subscriber::EnvFilter;

use hypervisor::AegisHypervisor;
use hal::GpuHal;

#[tokio::main]
async fn main() -> Result<()> {
    // ── Logging ──────────────────────────────────────────────────────────────
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info,aegis_gpu=debug")),
        )
        .with_target(true)
        .with_thread_ids(true)
        .init();

    info!("Aegis-GPU hypervisor starting — PID {}", std::process::id());

    // ── Hardware Abstraction Layer ────────────────────────────────────────────
    let hal = GpuHal::probe().map_err(|e| {
        error!("HAL probe failed: {e}");
        e
    })?;

    info!(
        device = %hal.device_name(),
        bar0_size = hal.bar0_size(),
        sm_count = hal.sm_count(),
        "GPU device enumerated"
    );

    // ── Hypervisor Bootstrap ─────────────────────────────────────────────────
    let hypervisor = AegisHypervisor::new(hal).await?;
    info!("Hypervisor initialised — waiting for guest connections");

    // ── IPC Server (Unix-domain socket for guest libraries) ──────────────────
    let ipc_server = ipc::IpcServer::bind("/run/aegis-gpu.sock")?;
    hypervisor.run(ipc_server).await?;

    info!("Aegis-GPU hypervisor shut down cleanly");
    Ok(())
}
