use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let workspace = manifest_dir.join("../..");
    let kernels_dir = workspace.join("kernels");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUTLASS_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=PATH");
    println!("cargo:rerun-if-changed={}", kernels_dir.display());

    register_tree_reruns(&kernels_dir, &["cu", "cuh"]);

    if !cuda_feature_enabled() {
        write_manifest(&out_dir, &[]);
        return;
    }

    let cu_files = collect_by_ext(&kernels_dir, "cu");

    let nvcc = find_nvcc().unwrap_or_else(|| {
        eprintln!(
            "\n\
beacon-cuda build failed: `nvcc` not found.\n\
\n\
The `cuda` feature is enabled but the CUDA toolkit was not detected.\n\
Install the CUDA toolkit and ensure `nvcc` is on PATH, or set one of:\n\
  CUDA_HOME=/usr/local/cuda\n\
  CUDA_PATH=/usr/local/cuda\n\
\n\
On vast.ai H100 boxes CUDA is usually preinstalled at /usr/local/cuda.\n\
Verify with: nvcc --version\n"
        );
        std::process::exit(1);
    });

    let cutlass = if cu_files.is_empty() {
        None
    } else {
        Some(resolve_cutlass(&workspace))
    };
    if let Some(ref cutlass) = cutlass {
        register_cutlass_reruns(cutlass);
    }

    let mut artifacts = Vec::new();
    for cu in &cu_files {
        let stem = cu
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_else(|| panic!("invalid kernel filename {}", cu.display()));
        let ptx_path = out_dir.join(format!("{stem}.ptx"));
        compile_ptx(
            &nvcc,
            cutlass.as_ref().expect("CUTLASS required when kernels exist"),
            &kernels_dir,
            cu,
            &ptx_path,
        );
        artifacts.push((stem.to_string(), ptx_path));
    }

    write_manifest(&out_dir, &artifacts);
}

fn cuda_feature_enabled() -> bool {
    env::var("CARGO_FEATURE_CUDA").is_ok()
}

fn find_nvcc() -> Option<PathBuf> {
    for key in ["CUDA_HOME", "CUDA_PATH", "CUDA_ROOT", "CUDA_TOOLKIT_ROOT_DIR"] {
        if let Ok(root) = env::var(key) {
            let candidate = PathBuf::from(root).join("bin/nvcc");
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }

    if let Ok(path) = env::var("PATH") {
        for dir in path.split(':') {
            let candidate = PathBuf::from(dir).join("nvcc");
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }

    let common = ["/usr/local/cuda/bin/nvcc", "/opt/cuda/bin/nvcc"];
    for candidate in common {
        let path = PathBuf::from(candidate);
        if path.is_file() {
            return Some(path);
        }
    }

    None
}

fn resolve_cutlass(workspace: &Path) -> PathBuf {
    if let Ok(path) = env::var("CUTLASS_PATH") {
        let path = PathBuf::from(path);
        if cutlass_is_valid(&path) {
            return path;
        }
        panic!(
            "CUTLASS_PATH is set to `{}` but `include/cute` was not found.\n\
Point CUTLASS_PATH at a CUTLASS checkout root, or clone into `third_party/cutlass`.",
            path.display()
        );
    }

    for candidate in [
        workspace.join("third_party/cutlass"),
        workspace.join("cutlass"),
    ] {
        if cutlass_is_valid(&candidate) {
            return candidate;
        }
    }

    panic!(
        "CUTLASS not found.\n\
Set CUTLASS_PATH to a CUTLASS checkout, or clone CUTLASS into:\n\
  {}\n\
Expected layout: <cutlass>/include/cute/...",
        workspace.join("third_party/cutlass").display()
    );
}

fn cutlass_is_valid(path: &Path) -> bool {
    path.join("include/cute").is_dir()
}

fn register_cutlass_reruns(cutlass: &Path) {
    println!("cargo:rerun-if-changed={}", cutlass.join("include").display());
    let git_head = cutlass.join(".git/HEAD");
    if git_head.exists() {
        println!("cargo:rerun-if-changed={}", git_head.display());
    }
    let git_file = cutlass.join(".git");
    if git_file.is_file() {
        if let Ok(contents) = fs::read_to_string(&git_file) {
            let git_dir = contents.trim().strip_prefix("gitdir: ").unwrap_or("");
            if !git_dir.is_empty() {
                let head = PathBuf::from(git_dir).join("HEAD");
                if head.exists() {
                    println!("cargo:rerun-if-changed={}", head.display());
                }
            }
        }
    }
}

fn register_tree_reruns(root: &Path, exts: &[&str]) {
    if !root.is_dir() {
        return;
    }
    register_tree_reruns_inner(root, exts);
}

fn register_tree_reruns_inner(dir: &Path, exts: &[&str]) {
    let entries = match fs::read_dir(dir) {
        Ok(v) => v,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            register_tree_reruns_inner(&path, exts);
            continue;
        }
        if has_ext(&path, exts) {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}

fn collect_by_ext(root: &Path, ext: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if !root.is_dir() {
        return out;
    }
    collect_by_ext_inner(root, ext, &mut out);
    out.sort();
    out
}

fn collect_by_ext_inner(dir: &Path, ext: &str, out: &mut Vec<PathBuf>) {
    let entries = match fs::read_dir(dir) {
        Ok(v) => v,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_by_ext_inner(&path, ext, out);
        } else if path.extension().and_then(|s| s.to_str()) == Some(ext) {
            out.push(path);
        }
    }
}

fn has_ext(path: &Path, exts: &[&str]) -> bool {
    let Some(ext) = path.extension().and_then(|s| s.to_str()) else {
        return false;
    };
    exts.iter().any(|e| *e == ext)
}

fn compile_ptx(nvcc: &Path, cutlass: &Path, kernels_dir: &Path, cu: &Path, ptx_out: &Path) {
    let cuda_include = cuda_include_dir();
    let mut cmd = Command::new(nvcc);
    cmd.arg("--ptx")
        .arg("-arch=sm_90a")
        .arg("-std=c++17")
        .arg(format!("-I{}", cutlass.join("include").display()))
        .arg(format!("-I{}", kernels_dir.display()))
        .arg("-o")
        .arg(ptx_out)
        .arg(cu);
    if let Some(include) = cuda_include {
        cmd.arg(format!("-I{include}"));
    }

    let output = cmd.output().unwrap_or_else(|err| {
        panic!("failed to spawn `{}`: {err}", nvcc.display());
    });

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        panic!(
            "nvcc failed compiling `{}` to `{}`\nstdout:\n{stdout}\nstderr:\n{stderr}",
            cu.display(),
            ptx_out.display()
        );
    }
}

fn cuda_include_dir() -> Option<String> {
    for key in ["CUDA_HOME", "CUDA_PATH", "CUDA_ROOT", "CUDA_TOOLKIT_ROOT_DIR"] {
        if let Ok(root) = env::var(key) {
            let include = PathBuf::from(root).join("include");
            if include.is_dir() {
                return Some(include.display().to_string());
            }
        }
    }
    let fallback = PathBuf::from("/usr/local/cuda/include");
    if fallback.is_dir() {
        return Some(fallback.display().to_string());
    }
    None
}

fn write_manifest(out_dir: &Path, artifacts: &[(String, PathBuf)]) {
    let path = out_dir.join("ptx_manifest.rs");
    let mut file = fs::File::create(&path).expect("create ptx_manifest.rs");

    writeln!(file, "pub struct PtxArtifact {{").unwrap();
    writeln!(file, "    pub name: &'static str,").unwrap();
    writeln!(file, "    pub path: &'static str,").unwrap();
    writeln!(file, "}}").unwrap();
    writeln!(file).unwrap();
    writeln!(file, "pub const PTX_ARTIFACTS: &[PtxArtifact] = &[").unwrap();
    for (name, ptx_path) in artifacts {
        let abs = fs::canonicalize(ptx_path).unwrap_or_else(|err| {
            panic!("failed to canonicalize {}: {err}", ptx_path.display());
        });
        writeln!(
            file,
            "    PtxArtifact {{ name: \"{name}\", path: \"{}\" }},",
            abs.display()
        )
        .unwrap();
    }
    writeln!(file, "];").unwrap();
    writeln!(file).unwrap();
    writeln!(file, "pub fn ptx_path(name: &str) -> Option<&'static str> {{").unwrap();
    writeln!(
        file,
        "    PTX_ARTIFACTS.iter().find(|a| a.name == name).map(|a| a.path)"
    )
    .unwrap();
    writeln!(file, "}}").unwrap();
}
