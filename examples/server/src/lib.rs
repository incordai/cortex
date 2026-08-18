#![recursion_limit = "141"]

use cortex::{server::Channel, tensor::Device};

pub fn start() {
    let port = std::env::var("REMOTE_BACKEND_PORT")
        .map(|port| match port.parse::<u16>() {
            Ok(val) => val,
            Err(err) => panic!("Invalid port, got {port} with error {err}"),
        })
        .unwrap_or(3000);

    cortex::server::start(Device::default(), Channel::WebSocket { port });
}
