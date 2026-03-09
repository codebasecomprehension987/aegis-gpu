// aegis-gpu/src/lib.rs
// Crate root — re-exports all public subsystems.

#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::pedantic, clippy::nursery)]

pub mod hal;
pub mod hypervisor;
pub mod ipc;
pub mod memory;
pub mod ptx_rewriter;
pub mod scheduler;
pub mod side_channel;
pub mod verification;
