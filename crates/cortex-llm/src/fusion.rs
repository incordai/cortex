//! [`LlmOps`] for the fusion decorator.
//!
//! Each call is registered as one custom operation and replayed through the
//! inner backend's `LlmOps`, so a CubeCL inner backend still runs the
//! dedicated kernels (`kernels/`) and the call sits in the fused stream like
//! any other op. Sibling pattern: `cortex-linalg`'s `ops.rs` (`linalg::svd`).

use alloc::{vec, vec::Vec};
use cortex_core::backend::{
    TensorMetadata,
    tensor::{FloatTensor, IntTensor},
};
use cortex_core::tensor::{DType, Shape};
use cortex_fusion::{
    ExecutionError, Fusion, FusionBackend, FusionRuntime,
    client::GlobalFusionClient,
    custom::{
        CustomOpIr, HandleContainer, Operation, OperationIr, OperationOutput, ScalarIr, StreamId,
        TensorIr,
    },
};

use crate::LlmOps;

#[derive(Debug, Clone, Copy)]
enum Kind {
    QuantizedMatmul,
    Dequantize,
    GatherQuantizedMatmul,
    BatchedQuantizedMatmul,
    RmsNorm,
    Rope,
}

#[derive(Debug)]
struct LlmOp<B> {
    desc: CustomOpIr,
    kind: Kind,
    _backend: core::marker::PhantomData<B>,
}

fn uint(s: &ScalarIr) -> u64 {
    match s {
        ScalarIr::UInt(v) => *v,
        other => panic!("llm custom op: expected an unsigned scalar, got {other:?}"),
    }
}

fn boolean(s: &ScalarIr) -> bool {
    match s {
        ScalarIr::Bool(v) => *v,
        other => panic!("llm custom op: expected a boolean scalar, got {other:?}"),
    }
}

fn float(s: &ScalarIr) -> f64 {
    match s {
        ScalarIr::Float(v) => *v,
        other => panic!("llm custom op: expected a float scalar, got {other:?}"),
    }
}

impl<B> Operation<B::FusionRuntime> for LlmOp<B>
where
    B: FusionBackend + LlmOps,
{
    fn execute(
        &self,
        handles: &mut HandleContainer<<B::FusionRuntime as FusionRuntime>::FusionHandle>,
    ) -> Result<(), ExecutionError> {
        let s = self.desc.scalars.as_slice();
        match self.kind {
            Kind::QuantizedMatmul => {
                let ([x, data, scales, biases], [out]) = self.desc.as_fixed();
                let y = B::quantized_matmul(
                    handles.get_float_tensor::<B>(x),
                    handles.get_int_tensor::<B>(data),
                    handles.get_float_tensor::<B>(scales),
                    handles.get_float_tensor::<B>(biases),
                    uint(&s[0]) as u32,
                    uint(&s[1]) as usize,
                    boolean(&s[2]),
                );
                handles.register_float_tensor::<B>(&out.id, y);
            }
            Kind::Dequantize => {
                let ([data, scales, biases], [out]) = self.desc.as_fixed();
                let w = B::dequantize_packed(
                    handles.get_int_tensor::<B>(data),
                    handles.get_float_tensor::<B>(scales),
                    handles.get_float_tensor::<B>(biases),
                    uint(&s[0]) as u32,
                    uint(&s[1]) as usize,
                    boolean(&s[2]),
                );
                handles.register_float_tensor::<B>(&out.id, w);
            }
            Kind::GatherQuantizedMatmul => {
                let ([x, data, scales, biases, indices], [out]) = self.desc.as_fixed();
                let y = B::gather_quantized_matmul(
                    handles.get_float_tensor::<B>(x),
                    handles.get_int_tensor::<B>(data),
                    handles.get_float_tensor::<B>(scales),
                    handles.get_float_tensor::<B>(biases),
                    handles.get_int_tensor::<B>(indices),
                    uint(&s[0]) as u32,
                    uint(&s[1]) as usize,
                    boolean(&s[2]),
                );
                handles.register_float_tensor::<B>(&out.id, y);
            }
            Kind::BatchedQuantizedMatmul => {
                let ([x, data, scales, biases], [out]) = self.desc.as_fixed();
                let y = B::batched_quantized_matmul(
                    handles.get_float_tensor::<B>(x),
                    handles.get_int_tensor::<B>(data),
                    handles.get_float_tensor::<B>(scales),
                    handles.get_float_tensor::<B>(biases),
                    uint(&s[0]) as usize,
                    uint(&s[1]) as u32,
                    uint(&s[2]) as usize,
                    boolean(&s[3]),
                    boolean(&s[4]),
                );
                handles.register_float_tensor::<B>(&out.id, y);
            }
            Kind::RmsNorm => {
                let ([x, gamma], [out]) = self.desc.as_fixed();
                let y = B::rms_norm(
                    handles.get_float_tensor::<B>(x),
                    handles.get_float_tensor::<B>(gamma),
                    float(&s[0]) as f32,
                    boolean(&s[1]),
                );
                handles.register_float_tensor::<B>(&out.id, y);
            }
            Kind::Rope => {
                let ([x, inv_freq], [out]) = self.desc.as_fixed();
                let y = B::rope(
                    handles.get_float_tensor::<B>(x),
                    handles.get_float_tensor::<B>(inv_freq),
                    uint(&s[0]) as usize,
                    boolean(&s[1]),
                );
                handles.register_float_tensor::<B>(&out.id, y);
            }
        }
        Ok(())
    }
}

fn register<B>(
    client: &GlobalFusionClient<B::FusionRuntime>,
    name: &'static str,
    kind: Kind,
    inputs: Vec<TensorIr>,
    out_shape: Shape,
    out_dtype: DType,
    scalars: Vec<ScalarIr>,
) -> FloatTensor<Fusion<B>>
where
    B: FusionBackend + LlmOps,
{
    let outputs = [TensorIr::uninit(client.create_empty_handle(), out_shape, out_dtype)];
    let desc = CustomOpIr::with_scalars(name, &inputs, &outputs, scalars);
    let [out] = client
        .register(
            StreamId::current(),
            OperationIr::Custom(desc.clone()),
            LlmOp::<B> {
                desc,
                kind,
                _backend: core::marker::PhantomData,
            },
        )
        .outputs();
    out
}

impl<B> LlmOps for Fusion<B>
where
    B: FusionBackend + LlmOps,
{
    fn quantized_matmul(
        x: FloatTensor<Self>,
        data: IntTensor<Self>,
        scales: FloatTensor<Self>,
        biases: FloatTensor<Self>,
        bits: u32,
        group_size: usize,
        affine: bool,
    ) -> FloatTensor<Self> {
        let client = x.client.clone();
        let dtype = x.dtype;
        let m = x.shape()[0];
        let n = data.shape()[1] * 32 / bits as usize;
        register::<B>(
            &client,
            "llm::quantized_matmul",
            Kind::QuantizedMatmul,
            vec![x.into_ir(), data.into_ir(), scales.into_ir(), biases.into_ir()],
            Shape::new([m, n]),
            dtype,
            vec![
                ScalarIr::UInt(bits as u64),
                ScalarIr::UInt(group_size as u64),
                ScalarIr::Bool(affine),
            ],
        )
    }

    fn dequantize_packed(
        data: IntTensor<Self>,
        scales: FloatTensor<Self>,
        biases: FloatTensor<Self>,
        bits: u32,
        group_size: usize,
        affine: bool,
    ) -> FloatTensor<Self> {
        let client = scales.client.clone();
        let dtype = scales.dtype;
        let d_shape = data.shape();
        let (rows, n) = (d_shape[0], d_shape[1] * 32 / bits as usize);
        register::<B>(
            &client,
            "llm::dequantize",
            Kind::Dequantize,
            vec![data.into_ir(), scales.into_ir(), biases.into_ir()],
            Shape::new([rows, n]),
            dtype,
            vec![
                ScalarIr::UInt(bits as u64),
                ScalarIr::UInt(group_size as u64),
                ScalarIr::Bool(affine),
            ],
        )
    }

    fn gather_quantized_matmul(
        x: FloatTensor<Self>,
        data: IntTensor<Self>,
        scales: FloatTensor<Self>,
        biases: FloatTensor<Self>,
        indices: IntTensor<Self>,
        bits: u32,
        group_size: usize,
        affine: bool,
    ) -> FloatTensor<Self> {
        let client = x.client.clone();
        let dtype = x.dtype;
        let t = x.shape()[0];
        let n = data.shape()[2] * 32 / bits as usize;
        register::<B>(
            &client,
            "llm::gather_quantized_matmul",
            Kind::GatherQuantizedMatmul,
            vec![
                x.into_ir(),
                data.into_ir(),
                scales.into_ir(),
                biases.into_ir(),
                indices.into_ir(),
            ],
            Shape::new([t, n]),
            dtype,
            vec![
                ScalarIr::UInt(bits as u64),
                ScalarIr::UInt(group_size as u64),
                ScalarIr::Bool(affine),
            ],
        )
    }

    fn batched_quantized_matmul(
        x: FloatTensor<Self>,
        data: IntTensor<Self>,
        scales: FloatTensor<Self>,
        biases: FloatTensor<Self>,
        rows: usize,
        bits: u32,
        group_size: usize,
        affine: bool,
        transpose: bool,
    ) -> FloatTensor<Self> {
        let client = x.client.clone();
        let dtype = x.dtype;
        let x_shape = x.shape();
        let n = if transpose { rows } else { data.shape()[2] * 32 / bits as usize };
        register::<B>(
            &client,
            "llm::batched_quantized_matmul",
            Kind::BatchedQuantizedMatmul,
            vec![x.into_ir(), data.into_ir(), scales.into_ir(), biases.into_ir()],
            Shape::new([x_shape[0], x_shape[1], n]),
            dtype,
            vec![
                ScalarIr::UInt(rows as u64),
                ScalarIr::UInt(bits as u64),
                ScalarIr::UInt(group_size as u64),
                ScalarIr::Bool(affine),
                ScalarIr::Bool(transpose),
            ],
        )
    }

    fn rms_norm(
        x: FloatTensor<Self>,
        gamma: FloatTensor<Self>,
        eps: f32,
        unit_offset: bool,
    ) -> FloatTensor<Self> {
        let client = x.client.clone();
        let (shape, dtype) = (x.shape(), x.dtype);
        register::<B>(
            &client,
            "llm::rms_norm",
            Kind::RmsNorm,
            vec![x.into_ir(), gamma.into_ir()],
            shape,
            dtype,
            vec![ScalarIr::Float(eps as f64), ScalarIr::Bool(unit_offset)],
        )
    }

    fn rope(
        x: FloatTensor<Self>,
        inv_freq: FloatTensor<Self>,
        offset: usize,
        traditional: bool,
    ) -> FloatTensor<Self> {
        let client = x.client.clone();
        let (shape, dtype) = (x.shape(), x.dtype);
        register::<B>(
            &client,
            "llm::rope",
            Kind::Rope,
            vec![x.into_ir(), inv_freq.into_ir()],
            shape,
            dtype,
            vec![ScalarIr::UInt(offset as u64), ScalarIr::Bool(traditional)],
        )
    }
}
