use core::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LaunchError {
    Unimplemented(&'static str),
    InvalidLaunchConfig(String),
    ModuleLoad(String),
    FunctionNotFound(String),
    Cuda(String),
    Cublas(String),
    OutOfMemory { requested: usize, available: usize },
    ShapeMismatch { expected: String, found: String },
}

pub type LaunchResult<T> = Result<T, LaunchError>;

impl fmt::Display for LaunchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LaunchError::Unimplemented(what) => {
                write!(f, "unimplemented without the `cuda` feature: {what}")
            }
            LaunchError::InvalidLaunchConfig(msg) => write!(f, "invalid launch config: {msg}"),
            LaunchError::ModuleLoad(msg) => write!(f, "ptx module load failed: {msg}"),
            LaunchError::FunctionNotFound(name) => write!(f, "kernel function not found: {name}"),
            LaunchError::Cuda(msg) => write!(f, "cuda error: {msg}"),
            LaunchError::Cublas(msg) => write!(f, "cublas error: {msg}"),
            LaunchError::OutOfMemory {
                requested,
                available,
            } => write!(
                f,
                "out of device memory: requested {requested} bytes, {available} available"
            ),
            LaunchError::ShapeMismatch { expected, found } => {
                write!(f, "shape mismatch: expected {expected}, found {found}")
            }
        }
    }
}

impl std::error::Error for LaunchError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_messages() {
        assert_eq!(
            LaunchError::Unimplemented("rmsnorm_adjoint_fwd").to_string(),
            "unimplemented without the `cuda` feature: rmsnorm_adjoint_fwd"
        );
        assert_eq!(
            LaunchError::FunctionNotFound("attention_adjoint_bwd".into()).to_string(),
            "kernel function not found: attention_adjoint_bwd"
        );
        assert_eq!(
            LaunchError::OutOfMemory {
                requested: 1024,
                available: 512
            }
            .to_string(),
            "out of device memory: requested 1024 bytes, 512 available"
        );
    }

    #[test]
    fn result_alias_works() {
        fn ok() -> LaunchResult<u32> {
            Ok(7)
        }
        fn err() -> LaunchResult<u32> {
            Err(LaunchError::Cuda("boom".into()))
        }
        assert_eq!(ok().unwrap(), 7);
        assert!(err().is_err());
    }

    #[test]
    fn is_std_error() {
        fn takes_error(_: &dyn std::error::Error) {}
        takes_error(&LaunchError::Cublas("lt descriptor".into()));
    }
}
