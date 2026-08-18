extern crate alloc;

#[test]
fn test_cortexpack_no_std() {
    use cortex_no_std_tests::cortexpack;
    let device = Default::default();

    // Run all Cortexpack tests
    cortexpack::run_all_tests(&device);
}
