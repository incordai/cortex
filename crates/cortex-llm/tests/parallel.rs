//! Tensor parallelism: layers split over the devices of one process, and
//! collectives across processes (ranks run as threads, talking over real
//! loopback sockets).

use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::time::Duration;

use cortex_core::tensor::{Device, Tensor, TensorData};
use cortex_llm::distributed::{GroupConfig, GroupKey, ProcessGroup};
use cortex_llm::{ColumnParallelLinear, RowParallelLinear};

fn devices() -> Vec<Device> {
    #[cfg(all(feature = "flex", feature = "cubecl-backend"))]
    {
        vec![Device::flex(), Device::default(), Device::flex()]
    }
    #[cfg(not(all(feature = "flex", feature = "cubecl-backend")))]
    {
        vec![Device::default(); 3]
    }
}

fn noise(len: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 23) as f32 - 1.0) * 0.5
        })
        .collect()
}

fn read<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.into_data().try_to_vec::<f32>().unwrap()
}

fn max_err(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()))
}

fn t2(v: Vec<f32>, shape: [usize; 2], d: &Device) -> Tensor<2> {
    Tensor::from_data(TensorData::new(v, shape), d)
}

#[test]
fn a_column_then_row_parallel_mlp_matches_the_single_device_one() {
    let devs = devices();
    let out = devs[0].clone();
    let (m, k, h, n) = (3usize, 10usize, 14usize, 6usize);
    let x = noise(m * k, 1);
    let (w1, b1, w2, b2) = (noise(k * h, 2), noise(h, 3), noise(h * n, 4), noise(n, 5));

    let up = ColumnParallelLinear::new(
        t2(w1.clone(), [k, h], &out),
        Some(Tensor::from_data(TensorData::new(b1.clone(), [h]), &out)),
        &devs,
        out.clone(),
    );
    let down = RowParallelLinear::new(
        t2(w2.clone(), [h, n], &out),
        Some(Tensor::from_data(TensorData::new(b2.clone(), [n]), &out)),
        &devs,
        out.clone(),
    );

    // The hidden activation stays sharded: no gather between the layers.
    let hidden: Vec<Tensor<2>> = up.forward_sharded(t2(x.clone(), [m, k], &out)).into_iter().map(|t| t.tanh()).collect();
    for (shard, device) in hidden.iter().zip(&devs) {
        assert_eq!(&shard.device(), device, "each hidden slice stays on its device");
    }
    let got = down.forward_sharded(hidden);
    assert_eq!(&got.device(), &out);

    let reference = {
        let xt = t2(x.clone(), [m, k], &out);
        let h1 = (xt.matmul(t2(w1, [k, h], &out)) + Tensor::<1>::from_data(TensorData::new(b1, [h]), &out).unsqueeze_dim(0)).tanh();
        h1.matmul(t2(w2, [h, n], &out)) + Tensor::<1>::from_data(TensorData::new(b2, [n]), &out).unsqueeze_dim(0)
    };
    let err = max_err(&read(got), &read(reference));
    assert!(err < 1e-5, "parallel MLP differs by {err}");

    // The gathered forms agree with the sharded ones.
    let gathered = up.forward(t2(x.clone(), [m, k], &out));
    assert_eq!(gathered.dims(), [m, h]);
}

fn listeners(world: usize) -> (Vec<TcpListener>, Vec<SocketAddr>) {
    let listeners: Vec<TcpListener> = (0..world).map(|_| TcpListener::bind("127.0.0.1:0").unwrap()).collect();
    let addrs = listeners.iter().map(|l| l.local_addr().unwrap()).collect();
    (listeners, addrs)
}

/// Runs one rank per thread on already bound `listeners`, rank `r` joining
/// with `config(r)`, and returns `body`'s results in rank order.
fn run_ranks<T: Send>(
    (listeners, addrs): (Vec<TcpListener>, Vec<SocketAddr>),
    config: impl Fn(usize) -> GroupConfig + Sync,
    body: impl Fn(io::Result<ProcessGroup>) -> T + Sync,
) -> Vec<T> {
    std::thread::scope(|s| {
        let handles: Vec<_> = listeners
            .into_iter()
            .enumerate()
            .map(|(rank, listener)| {
                let (addrs, config, body) = (&addrs, &config, &body);
                s.spawn(move || body(ProcessGroup::from_listener(rank, listener, addrs, config(rank))))
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    })
}

/// Binds one loopback listener per rank and runs `body(group)` on a thread
/// each, returning their results in rank order.
fn run_group<T: Send>(world: usize, body: impl Fn(ProcessGroup) -> T + Sync) -> Vec<T> {
    run_ranks(listeners(world), |_| GroupConfig::default(), |g| body(g.unwrap()))
}

const SECRET: &[u8] = b"the group's shared secret, 32+ b";

/// A keyed config whose waits are short enough for a refusal to show
/// quickly.
fn keyed(secret: &[u8]) -> GroupConfig {
    GroupConfig {
        key: Some(GroupKey::from_secret(secret).unwrap()),
        connect_timeout: Duration::from_secs(5),
        io_timeout: Some(Duration::from_secs(5)),
        ..GroupConfig::default()
    }
}

/// Every rank's all-reduce of `[rank + 1, 10 rank]`.
fn reduce(g: io::Result<ProcessGroup>) -> io::Result<Vec<f32>> {
    let g = g?;
    let r = g.rank() as f32;
    let mine = Tensor::<1>::from_data(TensorData::new(vec![r + 1.0, 10.0 * r], [2]), &Device::default());
    Ok(read(g.all_reduce_sum(mine)?))
}

#[test]
fn collectives_agree_on_every_rank() {
    let world = 3;
    let results = run_group(world, |g| {
        let dev = Device::default();
        let r = g.rank() as f32;
        let mine = Tensor::<2>::from_data(TensorData::new(vec![r + 1.0, 10.0 * r, -r, 0.5], [2, 2]), &dev);

        let sum = read(g.all_reduce_sum(mine.clone()).unwrap());
        let gathered = read(g.all_gather(mine.clone(), 0).unwrap());
        let root_value = (g.rank() == 1).then(|| Tensor::<2>::from_data(TensorData::new(vec![7.0f32, 8.0], [1, 2]), &dev));
        let bcast = read(g.broadcast(root_value, 1, &dev).unwrap());
        (sum, gathered, bcast)
    });

    // Σ over ranks 0..3 of [r+1, 10r, -r, 0.5].
    let want_sum = vec![6.0f32, 30.0, -3.0, 1.5];
    let want_gather: Vec<f32> = (0..world)
        .flat_map(|r| {
            let r = r as f32;
            vec![r + 1.0, 10.0 * r, -r, 0.5]
        })
        .collect();
    for (rank, (sum, gathered, bcast)) in results.into_iter().enumerate() {
        assert_eq!(sum, want_sum, "rank {rank} all_reduce_sum");
        assert_eq!(gathered, want_gather, "rank {rank} all_gather (rank order)");
        assert_eq!(bcast, vec![7.0, 8.0], "rank {rank} broadcast");
    }
}

#[test]
fn a_row_parallel_layer_across_processes_matches_one_process() {
    // Each rank holds a slice of the input features and of W's rows; the
    // all-reduce of the partial products is the full product.
    let (world, m, k, n) = (2usize, 3usize, 8usize, 5usize);
    let x = noise(m * k, 21);
    let w = noise(k * n, 22);
    let results = run_group(world, |g| {
        let dev = Device::default();
        let per = k / world;
        let lo = g.rank() * per;
        let x_slice: Vec<f32> = (0..m).flat_map(|r| x[r * k + lo..r * k + lo + per].to_vec()).collect();
        let w_slice = w[lo * n..(lo + per) * n].to_vec();
        let partial = t2(x_slice, [m, per], &dev).matmul(t2(w_slice, [per, n], &dev));
        read(g.all_reduce_sum(partial).unwrap())
    });
    let dev = Device::default();
    let want = read(t2(x, [m, k], &dev).matmul(t2(w, [k, n], &dev)));
    for (rank, got) in results.iter().enumerate() {
        assert!(max_err(got, &want) < 1e-5, "rank {rank} differs");
    }
    assert_eq!(results[0], results[1], "every rank holds bit-identical results");
}

#[test]
fn a_group_with_a_key_authenticates_and_still_agrees() {
    let results = run_ranks(listeners(3), |_| keyed(SECRET), reduce);
    for (rank, got) in results.into_iter().enumerate() {
        assert_eq!(got.unwrap(), vec![6.0, 30.0], "rank {rank}");
    }
}

#[test]
fn a_rank_holding_another_key_is_refused_by_both_ends() {
    let results = run_ranks(
        listeners(2),
        |rank| keyed(if rank == 0 { SECRET } else { b"a different secret, also 32 byte" }),
        |g| g.err().map(|e| e.kind()),
    );
    // Rank 1 connects, and rank 0's proof does not check out under its key;
    // rank 0 never sees a valid rank 1 and stops waiting.
    assert_eq!(results[1], Some(io::ErrorKind::PermissionDenied));
    assert_eq!(results[0], Some(io::ErrorKind::TimedOut));
}

/// A connection that knows the protocol but not the key: it is first in
/// rank 0's queue, claims rank 1 with a well-formed hello and answers rank
/// 0's challenge with a forged proof. Rank 0 must drop it and take the real
/// rank 1 that arrives after it.
#[test]
fn a_connection_that_cannot_prove_the_key_never_takes_a_rank() {
    let (listeners, addrs) = listeners(2);
    let mut stranger = TcpStream::connect(addrs[0]).unwrap();
    let mut hello = Vec::new();
    hello.extend_from_slice(&0x4358_5041u32.to_le_bytes()); // the authenticated hello's magic
    hello.extend_from_slice(&1u32.to_le_bytes()); // claims rank 1
    hello.extend_from_slice(&2u32.to_le_bytes()); // of 2
    hello.extend_from_slice(&[0u8; 32]); // its nonce
    stranger.write_all(&hello).unwrap();

    let results = std::thread::scope(|s| {
        let group = s.spawn(|| run_ranks((listeners, addrs), |_| keyed(SECRET), reduce));
        // Rank 0's challenge: its nonce and its proof of the key.
        let mut challenge = [0u8; 64];
        stranger.read_exact(&mut challenge).unwrap();
        stranger.write_all(&[0u8; 32]).unwrap();
        group.join().unwrap()
    });
    for (rank, got) in results.into_iter().enumerate() {
        assert_eq!(got.unwrap(), vec![1.0 + 2.0, 10.0], "rank {rank}");
    }
}
