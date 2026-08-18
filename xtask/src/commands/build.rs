use std::collections::HashMap;

use tracel_xtask::prelude::{clap::ValueEnum, *};

use crate::{ARM_NO_ATOMIC_PTR_TARGET, ARM_TARGET, NO_STD_CRATES, WASM32_TARGET};

#[macros::extend_command_args(BuildCmdArgs, Target, None)]
pub struct CortexBuildCmdArgs {
    /// Build in CI mode which excludes unsupported crates.
    #[arg(long)]
    pub ci: bool,
}

pub(crate) fn handle_command(
    mut args: CortexBuildCmdArgs,
    env: Environment,
    context: Context,
) -> anyhow::Result<()> {
    match context {
        Context::NoStd => {
            [
                "Default",
                WASM32_TARGET,
                ARM_TARGET,
                ARM_NO_ATOMIC_PTR_TARGET,
            ]
            .iter()
            .try_for_each(|build_target| {
                let mut build_args = vec!["--no-default-features"];
                let mut env_vars = HashMap::new();
                if *build_target != "Default" {
                    build_args.extend(vec!["--target", *build_target]);
                }

                let mut crates = NO_STD_CRATES.to_vec();

                if *build_target == ARM_NO_ATOMIC_PTR_TARGET {
                    // Only build a subset of crates which require `portable_atomic_unsafe_assume_single_core`.
                    // Other crates build with `cortex-flex` (default_backend) which requires the `critical-section`
                    // feature (automatically enabled via `cortex-dispatch` when `not(target_has_atomic = "ptr")`),
                    // which is mutually exclusive with `portable_atomic_unsafe_assume_single_core` cfg.
                    crates = vec!["cortex-std", "cortex-backend", "cortex-ndarray"];
                    env_vars.insert(
                        "RUSTFLAGS",
                        "--cfg portable_atomic_unsafe_assume_single_core",
                    );
                }
                helpers::custom_crates_build(
                    crates,
                    build_args.clone(),
                    Some(env_vars.clone()),
                    None,
                    &format!("no-std with target {}", *build_target),
                )?;

                // Second pass for `thumbv6m-none-eabi`: crates with `critical-section` feature
                // enabled so `once_cell` (transitively pulled from `cortex-flex` -> `gemm`) uses
                // portable-atomic for CAS emulation.
                if *build_target == ARM_NO_ATOMIC_PTR_TARGET {
                    crates = NO_STD_CRATES.to_vec();
                    // Remove `cortex-autodiff` from building with the
                    // target `thumbv6m-none-eabi` as it requires enabling the
                    // `arbitrary_self_types` feature for the
                    // `clone_if_require_grad` method of
                    // `cortex-autodiff::graph::Node`.
                    crates.retain(|&v| {
                        v != "cortex-autodiff"
                            && v != "cortex-std"
                            && v != "cortex-ndarray"
                            && v != "cortex-backend"
                    });

                    helpers::custom_crates_build(
                        crates,
                        build_args,
                        None,
                        None,
                        &format!("no-std with target {} (critical-section)", *build_target),
                    )?;
                }

                anyhow::Ok(())
            })?;
            Ok(())
        }
        Context::Std => {
            if args.ci {
                // Exclude crates that are not supported on CI
                args.exclude.extend(vec![
                    "cortex-cuda".to_string(),
                    "cortex-rocm".to_string(),
                    "cortex-tch".to_string(),
                ]);
                if std::env::var("DISABLE_WGPU").is_ok() {
                    args.exclude.extend(vec!["cortex-wgpu".to_string()]);
                };
            }
            // Build workspace
            base_commands::build::handle_command(args.try_into().unwrap(), env, context)?;
            // Specific additional commands to test specific features
            // cortex-dataset
            helpers::custom_crates_build(
                vec!["cortex-dataset"],
                vec!["--all-features"],
                None,
                None,
                "std with all features",
            )?;
            Ok(())
        }
        Context::All => Context::value_variants()
            .iter()
            .filter(|ctx| **ctx != Context::All)
            .try_for_each(|ctx| {
                handle_command(
                    CortexBuildCmdArgs {
                        target: args.target.clone(),
                        exclude: args.exclude.clone(),
                        only: args.only.clone(),
                        ci: args.ci,
                        release: args.release,
                        features: args.features.clone(),
                        no_default_features: args.no_default_features,
                    },
                    env.clone(),
                    ctx.clone(),
                )
            }),
    }
}
