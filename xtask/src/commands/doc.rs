use tracel_xtask::prelude::*;

fn set_cortex_device(device: &str) {
    // SAFETY: This is called in a single-threaded context within the xtask before spawning
    // child processes.
    unsafe {
        std::env::set_var("CORTEX_DEVICE", device);
    }
}

pub(crate) fn handle_command(
    mut args: DocCmdArgs,
    env: Environment,
    ctx: Context,
) -> anyhow::Result<()> {
    // The doc examples execute, so they need a device. flex is lightweight and portable, so
    // it is always available on the CI runners.
    set_cortex_device("flex");

    // Neither backend can link on the portable documentation runner. This also applies to
    // doctests: workspace feature unification would otherwise enable the HIP/CUDA runtime in
    // shared crates and make their doctest binaries link against unavailable system libraries.
    args.exclude
        .extend(vec!["cortex-cuda".to_string(), "cortex-rocm".to_string()]);

    // Execute documentation command on workspace
    base_commands::doc::handle_command(args.clone(), env, ctx)?;

    // Specific additional commands to build other docs
    if args.get_command() == DocSubCommand::Build {
        // cortex-dataset
        build_helpers::custom_crates_doc_build(
            vec!["cortex-dataset"],
            vec!["--all-features"],
            None,
            None,
            "All features",
        )?;
    }
    Ok(())
}
