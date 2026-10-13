#![cfg(all(feature = "persistence", not(target_family = "wasm")))]
//! What a cache root must do across processes: an entry one process writes is
//! there for the next, and two processes on one root at once neither fail nor
//! damage the file.
//!
//! Each process is this test binary run again with `CHILD_ROLE` set, which
//! makes the ignored `child` test play one role and exit with its verdict.

use std::path::Path;
use std::process::{Child, Command};

use cubecl_environment::persistence::{Namespace, Store, StoreError, StoreOptions};

const CHILD_ROLE: &str = "CUBECL_PROCESSES_TEST_ROLE";
const CHILD_ROOT: &str = "CUBECL_PROCESSES_TEST_ROOT";
const ENTRIES: u32 = 64;

fn open() -> Store<String, u32> {
    Store::new(StoreOptions::new().storage(Namespace::scoped("processes", "v1")))
}

fn key(writer: &str, entry: u32) -> String {
    format!("{writer}-{entry}")
}

fn spawn(role: &str, root: &Path) -> Child {
    Command::new(std::env::current_exe().unwrap())
        .args(["child", "--exact", "--ignored", "--nocapture", "--test-threads=1"])
        .env(CHILD_ROLE, role)
        .env(CHILD_ROOT, root)
        .spawn()
        .unwrap()
}

fn run(role: &str, root: &Path) {
    let status = spawn(role, root).wait().unwrap();
    assert!(status.success(), "child '{role}' failed: {status}");
}

/// One role, run in its own process.
#[test]
#[ignore = "run by the other tests in a child process"]
fn child() {
    let Ok(role) = std::env::var(CHILD_ROLE) else {
        return;
    };
    cubecl_environment::environment::set_root(std::env::var(CHILD_ROOT).unwrap());
    let mut store = open();
    match role.as_str() {
        // Writes its entries, then lives until the file `release` appears so
        // another process can open the root while this one holds it.
        "write" | "hold" => {
            for entry in 0..ENTRIES {
                store.insert(key("first", entry), entry).unwrap();
            }
            if role == "hold" {
                let root = std::env::var(CHILD_ROOT).unwrap();
                std::fs::write(Path::new(&root).join("written"), b"").unwrap();
                let release = Path::new(&root).join("release");
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
                while !release.exists() {
                    assert!(std::time::Instant::now() < deadline, "never released");
                    std::thread::sleep(std::time::Duration::from_millis(20));
                }
            }
        }
        // Reads the first writer's entries, then adds its own. Beside a
        // process holding the root the file is open read-only here, so a
        // write may be refused as a backend error, which cubecl's caches log
        // and go on from; any other outcome is a failure.
        "read" => {
            for entry in 0..ENTRIES {
                let key = key("first", entry);
                assert_eq!(store.get(&key).copied(), Some(entry), "{key}");
            }
            for entry in 0..ENTRIES {
                match store.insert(key("second", entry), entry) {
                    Ok(()) | Err(StoreError::Backend { .. }) => {}
                    Err(other) => panic!("{other:?}"),
                }
            }
        }
        other => panic!("unknown role {other}"),
    }
}

/// The case that made every process tune its kernels again: the cache must
/// reach the file, so a later process starts with it.
#[test]
#[cfg_attr(miri, ignore)]
fn entries_reach_the_next_process() {
    let root = tempfile::tempdir().unwrap();
    run("write", root.path());
    run("read", root.path());
}

/// A second process while the first holds the root reads what the first
/// wrote without failing, and the file is intact, and writable again, for a
/// third once both are gone.
#[test]
#[cfg_attr(miri, ignore)]
fn a_held_root_is_read_by_another_process() {
    let root = tempfile::tempdir().unwrap();
    let mut holder = spawn("hold", root.path());
    let written = root.path().join("written");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    while !written.exists() {
        assert!(std::time::Instant::now() < deadline, "the holder never wrote");
        assert!(holder.try_wait().unwrap().is_none(), "the holder exited early");
        std::thread::sleep(std::time::Duration::from_millis(20));
    }

    let reader = spawn("read", root.path()).wait().unwrap();
    std::fs::write(root.path().join("release"), b"").unwrap();
    let held = holder.wait().unwrap();
    assert!(reader.success(), "reader beside the holder failed: {reader}");
    assert!(held.success(), "holder failed: {held}");

    run("read", root.path());
}
