use crate::device::Device;
use crate::error::{LaunchError, LaunchResult};
use crate::launcher::KernelId;
use crate::ptx::ptx_path;

#[cfg(feature = "cuda")]
use std::collections::HashMap;

#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use crate::cuda::{map_driver_error, CudaFunction, CudaModule, Ptx};

#[derive(Default)]
pub struct ModuleCache {
    #[cfg(feature = "cuda")]
    modules: HashMap<String, Arc<CudaModule>>,
    #[cfg(feature = "cuda")]
    functions: HashMap<(String, String), CudaFunction>,
}

impl ModuleCache {
    pub fn new() -> Self {
        Self::default()
    }

    #[cfg(feature = "cuda")]
    pub fn load_module(
        &mut self,
        device: &Device,
        module_name: &str,
    ) -> LaunchResult<Arc<CudaModule>> {
        if let Some(module) = self.modules.get(module_name) {
            return Ok(module.clone());
        }
        let path = ptx_path(module_name).ok_or_else(|| {
            LaunchError::ModuleLoad(format!("no ptx artifact named `{module_name}`"))
        })?;
        let src = std::fs::read_to_string(path).map_err(|err| {
            LaunchError::ModuleLoad(format!("read ptx `{path}`: {err}"))
        })?;
        let module = device
            .context()
            .load_module(Ptx::from_src(src))
            .map_err(map_driver_error)?;
        self.modules
            .insert(module_name.to_string(), module.clone());
        Ok(module)
    }

    #[cfg(not(feature = "cuda"))]
    pub fn load_module(&mut self, device: &Device, module_name: &str) -> LaunchResult<()> {
        let _ = (device, module_name);
        Err(LaunchError::Unimplemented("ptx module load"))
    }

    #[cfg(feature = "cuda")]
    pub fn function(
        &mut self,
        device: &Device,
        module_name: &str,
        function_name: &str,
    ) -> LaunchResult<CudaFunction> {
        let key = (module_name.to_string(), function_name.to_string());
        if let Some(function) = self.functions.get(&key) {
            return Ok(function.clone());
        }
        let module = self.load_module(device, module_name)?;
        let function = module.load_function(function_name).map_err(|err| {
            LaunchError::FunctionNotFound(format!(
                "{function_name} in module `{module_name}`: {err}"
            ))
        })?;
        self.functions.insert(key, function.clone());
        Ok(function)
    }

    #[cfg(feature = "cuda")]
    pub fn kernel(
        &mut self,
        device: &Device,
        module_name: &str,
        kernel: KernelId,
    ) -> LaunchResult<CudaFunction> {
        self.function(device, module_name, kernel.name())
    }

    pub fn loaded_modules(&self) -> usize {
        #[cfg(feature = "cuda")]
        {
            self.modules.len()
        }
        #[cfg(not(feature = "cuda"))]
        {
            0
        }
    }

    pub fn cached_functions(&self) -> usize {
        #[cfg(feature = "cuda")]
        {
            self.functions.len()
        }
        #[cfg(not(feature = "cuda"))]
        {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "cuda")]
    fn cuda_driver_available() -> bool {
        std::env::var("BEACON_CUDA_TEST").ok().as_deref() == Some("1")
    }

    fn device_or_skip() -> Option<Device> {
        #[cfg(feature = "cuda")]
        if !cuda_driver_available() {
            return None;
        }
        Device::new(0).ok()
    }

    #[test]
    fn host_stub_rejects_load_without_cuda_feature() {
        #[cfg(not(feature = "cuda"))]
        {
            let device = Device::new(0).unwrap();
            let mut cache = ModuleCache::new();
            assert!(cache
                .load_module(&device, "elementwise")
                .unwrap_err()
                .to_string()
                .contains("unimplemented"));
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn missing_ptx_is_module_load_error() {
        let Some(device) = device_or_skip() else {
            return;
        };
        let mut cache = ModuleCache::new();
        let err = cache
            .load_module(&device, "definitely_missing_module")
            .unwrap_err();
        assert!(matches!(err, LaunchError::ModuleLoad(_)));
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn cache_counts_stay_zero_until_load() {
        let Some(_device) = device_or_skip() else {
            return;
        };
        let cache = ModuleCache::new();
        assert_eq!(cache.loaded_modules(), 0);
        assert_eq!(cache.cached_functions(), 0);
    }
}
