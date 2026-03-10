// aegis-gpu/src/ipc/mod.rs
// Unix-domain socket IPC server — accepts connections from guest user-space
// libraries and deserialises command packets.

use std::path::Path;
use tokio::net::{UnixListener, UnixStream};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use anyhow::Result;
use thiserror::Error;
use tracing::{debug, warn};

// ── Errors ────────────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum IpcError {
    #[error("client disconnected")]
    Disconnected,
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("protocol error: {0}")]
    Protocol(String),
}

// ── Wire protocol ─────────────────────────────────────────────────────────────
// Frame format: [4B: magic 0xAE615AFE][4B: length in bytes][N bytes: payload]

const FRAME_MAGIC: u32 = 0xAE61_5AFE;
const MAX_FRAME_BYTES: u32 = 1 * 1024 * 1024; // 1 MiB

// ── IpcServer ─────────────────────────────────────────────────────────────────

pub struct IpcServer {
    listener: UnixListener,
}

impl IpcServer {
    pub fn bind(path: impl AsRef<Path>) -> Result<Self> {
        let p = path.as_ref();
        if p.exists() { std::fs::remove_file(p)?; }
        let listener = UnixListener::bind(p)?;
        debug!("IPC listening on {:?}", p);
        Ok(Self { listener })
    }

    pub async fn accept(&mut self) -> Result<GuestConnection> {
        let (stream, addr) = self.listener.accept().await?;
        debug!("IPC connection from {:?}", addr);
        Ok(GuestConnection::new(stream))
    }
}

// ── GuestConnection ───────────────────────────────────────────────────────────

pub struct GuestConnection {
    stream:            UnixStream,
    requested_vram_gb: u64,
    priority:          u8,
}

impl GuestConnection {
    fn new(stream: UnixStream) -> Self {
        Self { stream, requested_vram_gb: 4, priority: 8 }
    }

    pub fn requested_vram_bytes(&self) -> u64 {
        self.requested_vram_gb * 1024 * 1024 * 1024
    }

    pub fn priority(&self) -> u8 { self.priority }

    /// Read one framed packet from the guest.
    pub async fn recv_packet(&mut self) -> Result<Vec<u8>, IpcError> {
        let mut hdr = [0u8; 8];
        match self.stream.read_exact(&mut hdr).await {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                return Err(IpcError::Disconnected);
            }
            Err(e) => return Err(IpcError::Io(e)),
        }

        let magic  = u32::from_le_bytes(hdr[0..4].try_into().unwrap());
        let length = u32::from_le_bytes(hdr[4..8].try_into().unwrap());

        if magic != FRAME_MAGIC {
            return Err(IpcError::Protocol(format!("bad magic 0x{magic:08X}")));
        }
        if length > MAX_FRAME_BYTES {
            return Err(IpcError::Protocol(format!("frame too large: {length} bytes")));
        }

        let mut payload = vec![0u8; length as usize];
        self.stream.read_exact(&mut payload).await
            .map_err(IpcError::Io)?;

        Ok(payload)
    }

    /// Send an acknowledgement or error code back to the guest library.
    pub async fn send_ack(&mut self, code: u32) -> Result<(), IpcError> {
        let frame = {
            let mut f = Vec::with_capacity(12);
            f.extend_from_slice(&FRAME_MAGIC.to_le_bytes());
            f.extend_from_slice(&4u32.to_le_bytes());   // length = 4
            f.extend_from_slice(&code.to_le_bytes());
            f
        };
        self.stream.write_all(&frame).await.map_err(IpcError::Io)
    }
}
