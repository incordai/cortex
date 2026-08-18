//! Client-side helper for registering [custom operations](CustomOpIr) on the remote backend.
//!
//! A backend extension ships its op to the server as `OperationIr::Custom`, where a registered
//! [`CustomOpRegistry`](cortex_router::CustomOpRegistry) handler executes it. How the op is registered
//! on the client differs by whether the `fusion` feature is enabled — the remote backend is a plain
//! [`BackendRouter`](cortex_router::BackendRouter) without it and a
//! [`Fusion`](cortex_fusion::Fusion)-wrapped one with it — and so does the tensor primitive it returns
//! (`RouterTensor` vs `FusionTensor`). [`CustomOpClient`] hides that difference: the same code builds
//! and registers a custom op, and gets back `FloatTensor<RemoteBackend>` either way.

use cortex_backend::tensor::FloatTensor;
use cortex_ir::{CustomOpIr, OperationIr, TensorId};

use crate::client::RemoteChannel;
use crate::{RemoteBackend, RemoteDevice};

/// The router channel used by the remote backend. Transport-erased, so it carries custom ops over
/// whichever transport the device was opened on.
type Channel = RemoteChannel;

/// A client for registering [custom operations](CustomOpIr) on the remote backend, transparently
/// across the `fusion` feature.
///
/// Drop-in for the lower-level client a backend extension would otherwise reach for: allocate output
/// ids with [`create_empty_handle`](Self::create_empty_handle), build a [`CustomOpIr`], then
/// [`register`](Self::register) it. With `fusion` enabled the op joins the cached op-graph (via the
/// fusion client); without it the op streams through the router client. Either way the op reaches the
/// server, and the returned tensors are the matching `FloatTensor<RemoteBackend>`.
pub struct CustomOpClient {
    #[cfg(not(feature = "fusion"))]
    inner: <Channel as cortex_router::RouterChannel>::Client,
    #[cfg(feature = "fusion")]
    inner: cortex_fusion::client::GlobalFusionClient<cortex_router::RouterFusionRuntime<Channel>>,
    /// Kept so the fusion path can build the op's unfused handler, which needs the device to reach
    /// the router client when the op isn't fused into a graph.
    #[cfg(feature = "fusion")]
    device: RemoteDevice,
}

impl CustomOpClient {
    /// Create a client bound to the given remote device.
    pub fn new(device: &RemoteDevice) -> Self {
        #[cfg(not(feature = "fusion"))]
        let inner = cortex_router::get_client::<Channel>(device);
        #[cfg(feature = "fusion")]
        let inner = cortex_fusion::get_client::<cortex_router::BackendRouter<Channel>>(device);

        Self {
            inner,
            #[cfg(feature = "fusion")]
            device: device.clone(),
        }
    }

    /// Allocate a fresh, uninitialized tensor id for a custom op output.
    ///
    /// Use it to build the output [`TensorIr`](cortex_ir::TensorIr)s of the [`CustomOpIr`] passed to
    /// [`register`](Self::register). The id is allocated by the same client that registers the op, so
    /// it is consistent with how that client tracks tensors (fusion ids under `fusion`, router ids
    /// otherwise).
    pub fn create_empty_handle(&self) -> TensorId {
        #[cfg(not(feature = "fusion"))]
        {
            use cortex_router::RouterClient;
            self.inner.create_empty_handle()
        }
        #[cfg(feature = "fusion")]
        {
            self.inner.create_empty_handle()
        }
    }

    /// Register a custom op and return its (uninitialized) output tensors.
    pub fn register(&self, op: CustomOpIr) -> Vec<FloatTensor<RemoteBackend>> {
        #[cfg(not(feature = "fusion"))]
        {
            use cortex_router::RouterClient;
            self.inner.register(OperationIr::Custom(op))
        }
        #[cfg(feature = "fusion")]
        {
            use cortex_fusion::stream::StreamId;
            use cortex_router::CustomOperation;
            // The op runs on the server through the `CustomOpRegistry`. When it's fused into a
            // cached op-graph it travels as part of the graph; when it isn't (a one-op segment),
            // `CustomOperation` is the unfused fallback that ships it through the router client.
            self.inner.register(
                StreamId::current(),
                OperationIr::Custom(op.clone()),
                CustomOperation::<Channel>::new(op, self.device.clone()),
            )
        }
    }
}
