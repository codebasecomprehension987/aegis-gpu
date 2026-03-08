// aegis-gpu/build.rs
// Compiles the CUDA scrub kernel to a fatbin and embeds it in the binary.
// Also sets cfg flags for SIMD and target-feature detection.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/side_channel/aegis_scrub.cu");
    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let fatbin  = out_dir.join("aegis_scrub.fatbin");

    // ── Compile CUDA kernel if nvcc is available ───────────────────────────
    if which_nvcc().is_some() {
        let status = Command::new("nvcc")
            .args([
                "--fatbin",
                "--generate-code", "arch=compute_80,code=sm_80",  // Ampere
                "--generate-code", "arch=compute_86,code=sm_86",  // Ampere LP
                "--generate-code", "arch=compute_90,code=sm_90",  // Hopper
                "-O3",
                "--use_fast_math",
                "--extra-device-vectorization",
                "-o", fatbin.to_str().unwrap(),
                "src/side_channel/aegis_scrub.cu",
            ])
            .status()
            .expect("nvcc failed to execute");

        if !status.success() {
            eprintln!("cargo:warning=nvcc compilation failed — using empty scrub stub");
            create_empty_fatbin(&fatbin);
        } else {
            println!("cargo:warning=Scrub kernel compiled to {:?}", fatbin);
        }
    } else {
        eprintln!("cargo:warning=nvcc not found — using empty scrub stub (simulation mode)");
        create_empty_fatbin(&fatbin);
    }

    // ── Target-feature detection ───────────────────────────────────────────
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected("avx512f") {
            println!("cargo:rustc-cfg=feature=\"simd-avx512\"");
        }
    }
}

fn which_nvcc() -> Option<PathBuf> {
    which::which("nvcc").ok()
}

fn create_empty_fatbin(path: &PathBuf) {
    std::fs::write(path, b"").unwrap();
}
