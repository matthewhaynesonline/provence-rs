//! This library provides shims for candle functions that are private, unimplemented, etc.
pub mod candle_transformers {
    pub mod models {
        pub mod debertav2 {
            use std::collections::HashMap;

            use candle_core::{Result, bail};
            use candle_transformers::models::debertav2::Config;

            pub fn id2label_len(
                config: &Config,
                id2label: Option<HashMap<u32, String>>,
            ) -> Result<usize> {
                let id2label_len = match (&config.id2label, id2label) {
                    (None, None) => bail!(
                        "Id2Label is either not present in the model configuration or not passed into DebertaV2NERModel::load as a parameter"
                    ),
                    (None, Some(id2label_p)) => id2label_p.len(),
                    (Some(id2label_c), None) => id2label_c.len(),
                    (Some(id2label_c), Some(id2label_p)) => {
                        if *id2label_c == id2label_p {
                            id2label_c.len()
                        } else {
                            bail!(
                                "Id2Label is both present in the model configuration and provided as a parameter, and they are different."
                            )
                        }
                    }
                };
                Ok(id2label_len)
            }
        }
    }
}

pub mod utils {
    pub mod device {
        //! Device management utilities.

        use candle_core::{Device, Result};

        /// Helper function to get a single candle Device. This will not perform any multi device mapping.
        /// Will prioritize Metal or CUDA (if Candle is compiled with those features) and fallback to CPU.
        ///
        /// # Arguments
        /// * `use_cpu` - Force CPU usage even if GPU is available
        /// * `quiet` - Suppress informational messages about GPU availability
        ///
        /// # Example  
        /// ```
        /// let device = candle_utils::device::get_device(true, false).unwrap();
        /// ```
        pub fn get_device(use_cpu: bool, quiet: bool) -> Result<Device> {
            if use_cpu {
                Ok(Device::Cpu)
            } else if candle_core::utils::cuda_is_available() {
                Ok(Device::new_cuda(0)?)
            } else if candle_core::utils::metal_is_available() {
                Ok(Device::new_metal(0)?)
            } else {
                if !quiet {
                    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
                    {
                        println!(
                            "Running on CPU, to run on GPU (metal), build with `--features metal`"
                        );
                    }

                    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
                    {
                        println!("Running on CPU, to run on GPU, build with `--features cuda`");
                    }
                }

                Ok(Device::Cpu)
            }
        }
    }
}
