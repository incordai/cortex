use super::optimization::ElemwiseOptimization;
use crate::{
    engine::{
        fuser::TraceOperationFuser,
        settings::{FuseSettings, RefLayoutSetting, VectorizationSetting},
    },
    optim::CubeOptimization,
};
use cortex_fusion::OperationFuser;
use cortex_std::Shape;
use cubecl::Runtime;

/// Fuses element wise operations.
pub struct ElementWiseFuser<R: Runtime> {
    fuser: TraceOperationFuser,
    device: R::Device,
}

impl<R: Runtime> Clone for ElementWiseFuser<R> {
    fn clone(&self) -> Self {
        Self {
            fuser: self.fuser.clone(),
            device: self.device.clone(),
        }
    }
}

impl<R: Runtime> ElementWiseFuser<R> {
    pub fn shape_id(&self) -> Shape {
        self.fuser.current_output_shape.clone()
    }
    pub fn new(device: R::Device) -> Self {
        let client = R::client(&device);
        let props = client.properties();
        let max_bindings = props.hardware.max_bindings;

        Self {
            fuser: TraceOperationFuser::new(
                max_bindings,
                FuseSettings {
                    broadcast: true,
                    output_shape_updates: true,
                    inplace: true,
                    vectorization: VectorizationSetting::Activated,
                    ref_layout: RefLayoutSetting::Any,
                },
            ),
            device,
        }
    }
}

impl<R: Runtime> OperationFuser<CubeOptimization<R>> for ElementWiseFuser<R> {
    fn fuse(&mut self, operation: &cortex_ir::OperationIr) {
        self.fuser.fuse(operation);
    }

    fn finish(&mut self) -> CubeOptimization<R> {
        let client = R::client(&self.device);
        let trace = self.fuser.finish();
        let elementwise = ElemwiseOptimization::new(trace, client, self.device.clone(), self.len());

        CubeOptimization::new(elementwise)
    }

    fn reset(&mut self) {
        self.fuser.reset()
    }

    fn status(&self) -> cortex_fusion::FuserStatus {
        self.fuser.status()
    }

    fn properties(&self) -> cortex_fusion::FuserProperties {
        self.fuser.properties()
    }

    fn len(&self) -> usize {
        self.fuser.len()
    }

    fn clone_dyn(&self) -> Box<dyn OperationFuser<CubeOptimization<R>>> {
        Box::new(self.clone())
    }
}
