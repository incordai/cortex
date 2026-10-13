//! Collectives across processes over TCP: `send` / `recv`, `broadcast`,
//! `all_gather` and `all_reduce_sum` on tensors of any backend.
//!
//! Every rank connects to every other (a full mesh) at start-up. Tensors
//! travel as their raw bytes with a small validated header, so this works
//! for every backend (the GPU ones read back to the host and upload again),
//! unlike the in-device collectives of `cortex-backend` (`DistributedOps`),
//! which need CUDA and NCCL. Exchanges run step by step around the ring of
//! ranks, each step sending on a scoped thread while receiving on the
//! current one, so a large tensor can never deadlock two ranks writing to
//! each other. Reductions sum in rank order, so every rank ends with
//! bit-identical results.
//!
//! With a [`GroupKey`] in its [`GroupConfig`], a group authenticates every
//! link: the two ends prove to each other that they hold the key in a
//! challenge–response handshake over fresh random nonces, and every frame
//! then carries keyed-hash tags over its header and its payload, bound to
//! the link, the direction and the frame's position in the stream. A frame
//! cannot then be forged, altered, replayed, reordered or reflected, and a
//! process without the key never becomes a member. Frames are not
//! encrypted: tensors still cross the network in the clear, so data that is
//! itself confidential needs a private network or a tunnel as well.
//!
//! Without a key a group is for a trusted network (one host, or a private
//! cluster link). Either way every frame is validated — magic, dtype, rank,
//! sizes — and capped in size before anything is allocated.
//!
//! Callers: model code running tensor or pipeline parallelism across
//! processes. Siblings: `parallel.rs` (the same layer splits across the
//! devices of one process), `cortex-backend`'s `DistributedOps` (NCCL).

use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::Mutex;
use std::time::{Duration, Instant};
use std::vec::Vec;

use cortex_core::tensor::{DType, Device, Tensor, TensorData};
use rand::TryRng;
use rand::rngs::SysRng;

const MAGIC: u32 = 0x4358_5047; // "CXPG"
/// The hello of a group that authenticates its links.
const MAGIC_AUTH: u32 = 0x4358_5041; // "CXPA"
const MAX_DIMS: usize = 8;
const HELLO: usize = 12;
const NONCE: usize = 32;
const TAG: usize = blake3::OUT_LEN;
/// How long an accepted connection may take over its handshake. A real
/// peer sends each part at once, so this only bounds a stray connection.
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);

/// A process group's pre-shared key: every rank must be given the same one.
///
/// Its `Debug` output hides the key.
#[derive(Clone)]
pub struct GroupKey([u8; blake3::KEY_LEN]);

impl GroupKey {
    /// The fewest secret bytes [`from_secret`](Self::from_secret) accepts.
    pub const MIN_SECRET: usize = 16;

    /// The key derived from `secret`, which every rank receives out of band
    /// (32 random bytes from a secrets store are ideal).
    ///
    /// # Errors
    /// If `secret` is shorter than [`MIN_SECRET`](Self::MIN_SECRET) bytes.
    pub fn from_secret(secret: &[u8]) -> io::Result<Self> {
        if secret.len() < Self::MIN_SECRET {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("a group key needs at least {} secret bytes", Self::MIN_SECRET),
            ));
        }
        Ok(Self(blake3::derive_key("cortex-llm 2026-10 process group key", secret)))
    }

    /// The keyed hash of `label` and `parts`, each part length-prefixed so
    /// no two different inputs hash alike.
    fn mac(&self, label: &[u8], parts: &[&[u8]]) -> blake3::Hash {
        let mut hasher = blake3::Hasher::new_keyed(&self.0);
        hasher.update(label);
        for part in parts {
            hasher.update(&(part.len() as u64).to_le_bytes());
            hasher.update(part);
        }
        hasher.finalize()
    }
}

impl core::fmt::Debug for GroupKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("GroupKey(..)")
    }
}

/// Settings of a [`ProcessGroup`].
#[derive(Clone, Debug)]
pub struct GroupConfig {
    /// How long to keep retrying connections to peers that are not up yet,
    /// and how long to wait for the higher ranks to connect in turn.
    pub connect_timeout: Duration,
    /// Read/write timeout on established connections; `None` waits forever.
    pub io_timeout: Option<Duration>,
    /// The largest tensor payload accepted, in bytes.
    pub max_message: usize,
    /// The key every link is authenticated with; `None` for a trusted
    /// network. Every rank must use the same setting.
    pub key: Option<GroupKey>,
}

impl Default for GroupConfig {
    fn default() -> Self {
        Self {
            connect_timeout: Duration::from_secs(60),
            io_timeout: Some(Duration::from_secs(600)),
            max_message: 1 << 32,
            key: None,
        }
    }
}

fn invalid(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

fn forged() -> io::Error {
    invalid("frame failed authentication")
}

fn dtype_code(dtype: DType) -> io::Result<u8> {
    Ok(match dtype {
        DType::F32 => 0,
        DType::F16 => 1,
        DType::BF16 => 2,
        DType::F64 => 3,
        other => return Err(io::Error::new(io::ErrorKind::Unsupported, format!("cannot send {other:?} tensors"))),
    })
}

fn dtype_from(code: u8) -> io::Result<DType> {
    Ok(match code {
        0 => DType::F32,
        1 => DType::F16,
        2 => DType::BF16,
        3 => DType::F64,
        other => return Err(invalid(format!("unknown dtype code {other}"))),
    })
}

/// One direction of an authenticated link: the key its frames are tagged
/// with, and the number of the next frame, which every tag covers.
pub(crate) struct FrameKey {
    key: [u8; blake3::KEY_LEN],
    seq: u64,
}

impl FrameKey {
    fn hasher(&self, part: &[u8]) -> blake3::Hasher {
        let mut hasher = blake3::Hasher::new_keyed(&self.key);
        hasher.update(&self.seq.to_le_bytes());
        hasher.update(part);
        hasher
    }

    fn head_tag(&self, head: &[u8]) -> blake3::Hash {
        self.hasher(b"head").update(head).finalize()
    }

    fn body_tag(&self, head: &[u8], payload: &[u8]) -> blake3::Hash {
        self.hasher(b"body").update(head).update(payload).finalize()
    }
}

fn read_tag(r: &mut impl Read) -> io::Result<blake3::Hash> {
    let mut tag = [0u8; TAG];
    r.read_exact(&mut tag)?;
    Ok(blake3::Hash::from(tag))
}

/// Writes one tensor frame: magic, dtype, rank count, dims, byte length,
/// bytes. With a `seal` key, a tag follows the header and another the
/// payload.
pub(crate) fn write_frame(w: &mut impl Write, data: &TensorData, seal: Option<&mut FrameKey>) -> io::Result<()> {
    let dims = data.shape().to_vec();
    if dims.len() > MAX_DIMS {
        return Err(invalid(format!("{} dimensions exceed the limit of {MAX_DIMS}", dims.len())));
    }
    let bytes = data.as_bytes();
    let mut head = Vec::with_capacity(16 + 8 * dims.len());
    head.extend_from_slice(&MAGIC.to_le_bytes());
    head.push(dtype_code(data.dtype())?);
    head.push(dims.len() as u8);
    for d in &dims {
        head.extend_from_slice(&(*d as u64).to_le_bytes());
    }
    head.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    w.write_all(&head)?;
    match seal {
        None => w.write_all(bytes)?,
        Some(key) => {
            w.write_all(key.head_tag(&head).as_bytes())?;
            w.write_all(bytes)?;
            w.write_all(key.body_tag(&head, bytes).as_bytes())?;
            key.seq += 1;
        }
    }
    w.flush()
}

/// Reads one tensor frame, refusing anything malformed or larger than
/// `max_message` before allocating for it. With an `open` key, the header's
/// tag is checked before its sizes are trusted and the payload's before it
/// is returned.
pub(crate) fn read_frame(r: &mut impl Read, max_message: usize, open: Option<&mut FrameKey>) -> io::Result<TensorData> {
    let mut fixed = [0u8; 6];
    r.read_exact(&mut fixed)?;
    if u32::from_le_bytes([fixed[0], fixed[1], fixed[2], fixed[3]]) != MAGIC {
        return Err(invalid("bad frame magic"));
    }
    let rank = fixed[5] as usize;
    if rank > MAX_DIMS {
        return Err(invalid(format!("{rank} dimensions exceed the limit of {MAX_DIMS}")));
    }
    let mut head = vec![0u8; 6 + 8 * rank + 8];
    head[..6].copy_from_slice(&fixed);
    r.read_exact(&mut head[6..])?;
    if let Some(key) = &open
        && read_tag(r)? != key.head_tag(&head)
    {
        return Err(forged());
    }

    let dtype = dtype_from(head[4])?;
    let field = |i: usize| u64::from_le_bytes(head[6 + 8 * i..14 + 8 * i].try_into().expect("8 bytes"));
    let mut dims = Vec::with_capacity(rank);
    let mut elems: usize = 1;
    for i in 0..rank {
        let d = usize::try_from(field(i)).map_err(|_| invalid("dimension too large"))?;
        elems = elems.checked_mul(d).ok_or_else(|| invalid("shape overflows"))?;
        dims.push(d);
    }
    let len = usize::try_from(field(rank)).map_err(|_| invalid("payload too large"))?;
    let expected = elems.checked_mul(dtype.size()).ok_or_else(|| invalid("shape overflows"))?;
    if len != expected {
        return Err(invalid(format!("payload is {len} bytes, the shape needs {expected}")));
    }
    if len > max_message {
        return Err(invalid(format!("payload of {len} bytes exceeds the limit of {max_message}")));
    }
    let mut bytes = vec![0u8; len];
    r.read_exact(&mut bytes)?;
    if let Some(key) = open {
        if read_tag(r)? != key.body_tag(&head, &bytes) {
            return Err(forged());
        }
        key.seq += 1;
    }
    Ok(TensorData::from_bytes_vec(bytes, dims, dtype))
}

/// One direction of a connection: its stream and, on an authenticated
/// link, that direction's frame key.
struct Half {
    stream: TcpStream,
    key: Option<FrameKey>,
}

/// One established connection, split into independent read and write
/// handles so a step can send and receive on it at once.
struct Link {
    reader: Mutex<Half>,
    writer: Mutex<Half>,
}

/// The frame keys of one link, for sending and for receiving.
struct LinkKeys {
    send: FrameKey,
    recv: FrameKey,
}

impl Link {
    fn new(stream: TcpStream, config: &GroupConfig, keys: Option<LinkKeys>) -> io::Result<Self> {
        stream.set_nodelay(true)?;
        stream.set_read_timeout(config.io_timeout)?;
        stream.set_write_timeout(config.io_timeout)?;
        let (send, recv) = match keys {
            Some(LinkKeys { send, recv }) => (Some(send), Some(recv)),
            None => (None, None),
        };
        Ok(Self {
            writer: Mutex::new(Half {
                stream: stream.try_clone()?,
                key: send,
            }),
            reader: Mutex::new(Half { stream, key: recv }),
        })
    }
}

/// A fixed set of processes that exchange tensors.
pub struct ProcessGroup {
    rank: usize,
    world: usize,
    /// `links[r]` is the connection to rank `r` (`None` for this rank).
    links: Vec<Option<Link>>,
    max_message: usize,
}

fn hello(magic: u32, rank: usize, world: usize) -> [u8; HELLO] {
    let mut b = [0u8; HELLO];
    b[..4].copy_from_slice(&magic.to_le_bytes());
    b[4..8].copy_from_slice(&(rank as u32).to_le_bytes());
    b[8..].copy_from_slice(&(world as u32).to_le_bytes());
    b
}

fn nonce() -> io::Result<[u8; NONCE]> {
    let mut nonce = [0u8; NONCE];
    SysRng
        .try_fill_bytes(&mut nonce)
        .map_err(|e| io::Error::other(format!("the OS random source failed: {e}")))?;
    Ok(nonce)
}

/// The frame keys of a link whose handshake was `transcript` (both hellos'
/// content and both nonces), as (connector to acceptor, acceptor to
/// connector). Fresh nonces make them unique to the link.
fn session_keys(key: &GroupKey, transcript: &[u8]) -> (FrameKey, FrameKey) {
    let derive = |label: &[u8]| FrameKey {
        key: *key.mac(label, &[transcript]).as_bytes(),
        seq: 0,
    };
    (derive(b"frames from the connector"), derive(b"frames from the acceptor"))
}

/// The connecting side's handshake with `peer`: sends its hello and a fresh
/// nonce, checks the acceptor's proof of the key over both nonces, then
/// proves its own.
fn connect_handshake(
    stream: &mut TcpStream,
    key: &GroupKey,
    rank: usize,
    world: usize,
    peer: usize,
    deadline: Instant,
) -> io::Result<LinkKeys> {
    let wait = deadline.saturating_duration_since(Instant::now()).max(Duration::from_secs(1));
    stream.set_read_timeout(Some(wait))?;
    let mut transcript = Vec::with_capacity(HELLO + 2 * NONCE);
    transcript.extend_from_slice(&hello(MAGIC_AUTH, rank, world));
    transcript.extend_from_slice(&nonce()?);
    stream.write_all(&transcript)?;

    let mut reply = [0u8; NONCE + TAG];
    stream.read_exact(&mut reply).map_err(|e| {
        io::Error::new(
            e.kind(),
            format!("rank {peer} ended the handshake ({e}); does it use the same group key?"),
        )
    })?;
    transcript.extend_from_slice(&reply[..NONCE]);
    let proof = blake3::Hash::from(<[u8; TAG]>::try_from(&reply[NONCE..]).expect("TAG bytes"));
    if proof != key.mac(b"acceptor proof", &[&transcript]) {
        return Err(io::Error::new(
            io::ErrorKind::PermissionDenied,
            format!("rank {peer} does not hold the group key"),
        ));
    }
    stream.write_all(key.mac(b"connector proof", &[&transcript]).as_bytes())?;
    let (send, recv) = session_keys(key, &transcript);
    Ok(LinkKeys { send, recv })
}

/// What the accepting side makes of a new connection.
enum Accepted {
    /// A peer of this group, with its link's keys when the group has a key.
    Peer(usize, Option<LinkKeys>),
    /// Anything else — a stray connection, another group, a duplicate, a
    /// process without the key — which is dropped: the expected peer may
    /// still arrive.
    Refused,
}

/// The accepting side's handshake on a new connection: reads the hello and
/// checks it names an expected higher rank of this group; with a key, also
/// answers the connector's nonce with its own and a proof of the key, then
/// requires the connector's proof over both nonces.
///
/// Only a failure of this process (its random source) is an error; any
/// failure of the connection refuses it.
fn accept_handshake(
    stream: &mut TcpStream,
    key: Option<&GroupKey>,
    rank: usize,
    world: usize,
    expected: impl Fn(usize) -> bool,
) -> io::Result<Accepted> {
    let mut theirs = [0u8; HELLO + NONCE];
    let hello_len = if key.is_some() { HELLO + NONCE } else { HELLO };
    if stream.read_exact(&mut theirs[..hello_len]).is_err() {
        return Ok(Accepted::Refused);
    }
    let field = |i: usize| u32::from_le_bytes(theirs[4 * i..4 * i + 4].try_into().expect("4 bytes"));
    let (magic, peer, their_world) = (field(0), field(1) as usize, field(2) as usize);
    let magic_wanted = if key.is_some() { MAGIC_AUTH } else { MAGIC };
    if magic != magic_wanted || their_world != world || peer <= rank || peer >= world || !expected(peer) {
        return Ok(Accepted::Refused);
    }
    let Some(key) = key else {
        return Ok(Accepted::Peer(peer, None));
    };

    let mut transcript = theirs.to_vec();
    let ours = nonce()?;
    transcript.extend_from_slice(&ours);
    let mut reply = Vec::with_capacity(NONCE + TAG);
    reply.extend_from_slice(&ours);
    reply.extend_from_slice(key.mac(b"acceptor proof", &[&transcript]).as_bytes());
    if stream.write_all(&reply).is_err() {
        return Ok(Accepted::Refused);
    }
    let Ok(proof) = read_tag(stream) else {
        return Ok(Accepted::Refused);
    };
    if proof != key.mac(b"connector proof", &[&transcript]) {
        return Ok(Accepted::Refused);
    }
    let (from_connector, from_acceptor) = session_keys(key, &transcript);
    Ok(Accepted::Peer(
        peer,
        Some(LinkKeys {
            send: from_acceptor,
            recv: from_connector,
        }),
    ))
}

impl ProcessGroup {
    /// Joins the group as `rank` of `addrs.len()`, listening on
    /// `addrs[rank]` and connecting to every other address.
    pub fn bind(rank: usize, addrs: &[SocketAddr], config: GroupConfig) -> io::Result<Self> {
        let listener = TcpListener::bind(addrs.get(rank).ok_or_else(|| invalid("rank out of range"))?)?;
        Self::from_listener(rank, listener, addrs, config)
    }

    /// As [`bind`](Self::bind), with this rank's listener already bound
    /// (useful to bind port 0 and publish the real port first).
    ///
    /// Rank `r` connects to every lower rank and accepts every higher one,
    /// so each pair gets exactly one connection.
    ///
    /// # Errors
    /// If a lower rank cannot be reached, or the higher ranks have not all
    /// joined, within `config.connect_timeout`; with a key, if a lower rank
    /// does not prove it holds the same key (`PermissionDenied`).
    pub fn from_listener(rank: usize, listener: TcpListener, addrs: &[SocketAddr], config: GroupConfig) -> io::Result<Self> {
        let world = addrs.len();
        if rank >= world {
            return Err(invalid(format!("rank {rank} of a group of {world}")));
        }
        let mut links: Vec<Option<Link>> = (0..world).map(|_| None).collect();

        let deadline = Instant::now() + config.connect_timeout;
        for (peer, addr) in addrs.iter().enumerate().take(rank) {
            let mut stream = loop {
                match TcpStream::connect_timeout(addr, Duration::from_secs(5)) {
                    Ok(s) => break s,
                    Err(e) if Instant::now() >= deadline => return Err(e),
                    Err(_) => std::thread::sleep(Duration::from_millis(50)),
                }
            };
            let keys = match &config.key {
                Some(key) => Some(connect_handshake(&mut stream, key, rank, world, peer, deadline)?),
                None => {
                    stream.write_all(&hello(MAGIC, rank, world))?;
                    None
                }
            };
            links[peer] = Some(Link::new(stream, &config, keys)?);
        }

        // Accepting polls, so a higher rank that never arrives (or never
        // proves the key) ends in an error rather than a hang.
        listener.set_nonblocking(true)?;
        let deadline = Instant::now() + config.connect_timeout;
        let mut pending = world - rank - 1;
        while pending > 0 {
            let mut stream = match listener.accept() {
                Ok((stream, _)) => stream,
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                    if Instant::now() >= deadline {
                        return Err(io::Error::new(
                            io::ErrorKind::TimedOut,
                            format!("{pending} higher rank(s) never joined rank {rank}"),
                        ));
                    }
                    std::thread::sleep(Duration::from_millis(10));
                    continue;
                }
                Err(e) => return Err(e),
            };
            stream.set_nonblocking(false)?;
            stream.set_read_timeout(Some(config.connect_timeout.min(HANDSHAKE_TIMEOUT)))?;
            stream.set_write_timeout(Some(config.connect_timeout.min(HANDSHAKE_TIMEOUT)))?;
            let unlinked = |peer: usize| links[peer].is_none();
            if let Accepted::Peer(peer, keys) = accept_handshake(&mut stream, config.key.as_ref(), rank, world, unlinked)? {
                links[peer] = Some(Link::new(stream, &config, keys)?);
                pending -= 1;
            }
        }

        Ok(Self {
            rank,
            world,
            links,
            max_message: config.max_message,
        })
    }

    /// This process's rank.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Number of processes.
    pub fn world(&self) -> usize {
        self.world
    }

    fn link(&self, peer: usize) -> io::Result<&Link> {
        self.links
            .get(peer)
            .and_then(Option::as_ref)
            .ok_or_else(|| invalid(format!("no connection to rank {peer}")))
    }

    fn send_data(&self, peer: usize, data: &TensorData) -> io::Result<()> {
        let link = self.link(peer)?;
        let mut half = link.writer.lock().map_err(|_| invalid("connection lock poisoned"))?;
        let Half { stream, key } = &mut *half;
        write_frame(stream, data, key.as_mut())
    }

    fn recv_data(&self, peer: usize) -> io::Result<TensorData> {
        let link = self.link(peer)?;
        let mut half = link.reader.lock().map_err(|_| invalid("connection lock poisoned"))?;
        let Half { stream, key } = &mut *half;
        read_frame(stream, self.max_message, key.as_mut())
    }

    /// Sends `tensor` to `peer`.
    pub fn send<const D: usize>(&self, peer: usize, tensor: Tensor<D>) -> io::Result<()> {
        self.send_data(peer, &tensor.into_data())
    }

    /// Receives a tensor from `peer` onto `device`.
    pub fn recv<const D: usize>(&self, peer: usize, device: &Device) -> io::Result<Tensor<D>> {
        upload(self.recv_data(peer)?, device)
    }

    /// Every rank's data, in rank order (this rank's own included), after
    /// one ring exchange.
    fn exchange(&self, mine: TensorData) -> io::Result<Vec<TensorData>> {
        let mut all: Vec<Option<TensorData>> = (0..self.world).map(|_| None).collect();
        for step in 1..self.world {
            let dst = (self.rank + step) % self.world;
            let src = (self.rank + self.world - step) % self.world;
            let received = std::thread::scope(|s| {
                let sender = s.spawn(|| self.send_data(dst, &mine));
                let got = self.recv_data(src);
                let sent = sender.join().map_err(|_| invalid("send thread panicked"))?;
                sent.and(got)
            })?;
            all[src] = Some(received);
        }
        all[self.rank] = Some(mine);
        Ok(all.into_iter().map(|d| d.expect("every rank exchanged")).collect())
    }

    /// Every rank's `tensor` concatenated along `dim`, in rank order.
    pub fn all_gather<const D: usize>(&self, tensor: Tensor<D>, dim: usize) -> io::Result<Tensor<D>> {
        let device = tensor.device();
        let parts = self
            .exchange(tensor.into_data())?
            .into_iter()
            .map(|d| upload(d, &device))
            .collect::<io::Result<Vec<Tensor<D>>>>()?;
        Ok(Tensor::cat(parts, dim))
    }

    /// The element-wise sum of every rank's `tensor`, identical on every rank.
    ///
    /// # Errors
    /// If the ranks' shapes differ, or a connection fails.
    pub fn all_reduce_sum<const D: usize>(&self, tensor: Tensor<D>) -> io::Result<Tensor<D>> {
        let device = tensor.device();
        let dtype = tensor.dtype();
        let shape = tensor.dims();
        let all = self.exchange(tensor.into_data())?;
        let mut sum: Vec<f64> = Vec::new();
        for data in all {
            if data.shape().as_slice() != shape.as_slice() {
                return Err(invalid("all_reduce_sum needs the same shape on every rank"));
            }
            let values = data
                .convert::<f64>()
                .try_into_vec::<f64>()
                .map_err(|e| invalid(format!("{e:?}")))?;
            if sum.is_empty() {
                sum = values;
            } else {
                sum.iter_mut().zip(values).for_each(|(s, v)| *s += v);
            }
        }
        let sum: Vec<f32> = sum.into_iter().map(|v| v as f32).collect();
        Ok(Tensor::<D>::from_data(TensorData::new(sum, shape), &device).cast(dtype))
    }

    /// `root`'s tensor on every rank. The root passes `Some(tensor)`, the
    /// others `None`.
    pub fn broadcast<const D: usize>(&self, tensor: Option<Tensor<D>>, root: usize, device: &Device) -> io::Result<Tensor<D>> {
        if self.rank == root {
            let tensor = tensor.ok_or_else(|| invalid("the root must pass its tensor"))?;
            let data = tensor.clone().into_data();
            for peer in (0..self.world).filter(|&p| p != root) {
                self.send_data(peer, &data)?;
            }
            Ok(tensor)
        } else {
            upload(self.recv_data(root)?, device)
        }
    }
}

fn upload<const D: usize>(data: TensorData, device: &Device) -> io::Result<Tensor<D>> {
    if data.shape().num_dims() != D {
        return Err(invalid(format!("received a rank-{} tensor, expected rank {D}", data.shape().num_dims())));
    }
    let dtype = data.dtype();
    Ok(Tensor::from_data(data, (device, dtype)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(data: &TensorData) -> Vec<u8> {
        let mut buf = Vec::new();
        write_frame(&mut buf, data, None).unwrap();
        buf
    }

    #[test]
    fn frames_round_trip() {
        let data = TensorData::new(vec![1.0f32, -2.5, 3.25, 0.0, 9.0, 7.0], [2, 3]);
        let back = read_frame(&mut frame(&data).as_slice(), 1 << 20, None).unwrap();
        assert_eq!(back.shape().to_vec(), vec![2, 3]);
        assert_eq!(back.try_to_vec::<f32>().unwrap(), vec![1.0, -2.5, 3.25, 0.0, 9.0, 7.0]);
    }

    #[test]
    fn malformed_frames_are_refused_before_allocating() {
        let data = TensorData::new(vec![1.0f32; 4], [4]);
        let good = frame(&data);

        let mut bad_magic = good.clone();
        bad_magic[0] ^= 0xFF;
        assert!(read_frame(&mut bad_magic.as_slice(), 1 << 20, None).is_err());

        // Byte length disagreeing with the shape.
        let mut bad_len = good.clone();
        let len_at = 6 + 8;
        bad_len[len_at..len_at + 8].copy_from_slice(&(1u64 << 40).to_le_bytes());
        let err = read_frame(&mut bad_len.as_slice(), usize::MAX, None).unwrap_err();
        assert!(err.to_string().contains("the shape needs"), "{err}");

        // Over the size cap.
        let err = read_frame(&mut good.as_slice(), 8, None).unwrap_err();
        assert!(err.to_string().contains("exceeds the limit"), "{err}");

        // A shape whose element count overflows.
        let mut huge = Vec::new();
        huge.extend_from_slice(&MAGIC.to_le_bytes());
        huge.extend_from_slice(&[0, 2]);
        huge.extend_from_slice(&u64::MAX.to_le_bytes());
        huge.extend_from_slice(&u64::MAX.to_le_bytes());
        huge.extend_from_slice(&0u64.to_le_bytes());
        assert!(read_frame(&mut huge.as_slice(), usize::MAX, None).is_err());
    }

    fn keys() -> (FrameKey, FrameKey) {
        let key = GroupKey::from_secret(b"sixteen+ bytes of secret").unwrap();
        session_keys(&key, b"a handshake transcript")
    }

    /// Two sealed frames, as one direction of a link sends them.
    fn sealed() -> (Vec<u8>, usize) {
        let (mut send, _) = keys();
        let mut wire = Vec::new();
        write_frame(&mut wire, &TensorData::new(vec![1.0f32, 2.0, 3.0], [3]), Some(&mut send)).unwrap();
        let first = wire.len();
        write_frame(&mut wire, &TensorData::new(vec![4.0f32, 5.0], [1, 2]), Some(&mut send)).unwrap();
        (wire, first)
    }

    fn read_all(wire: &[u8], mut recv: FrameKey) -> io::Result<Vec<Vec<f32>>> {
        let mut r = wire;
        let mut out = Vec::new();
        while !r.is_empty() {
            out.push(read_frame(&mut r, 1 << 20, Some(&mut recv))?.try_to_vec::<f32>().unwrap());
        }
        Ok(out)
    }

    #[test]
    fn sealed_frames_round_trip_in_order() {
        let (wire, _) = sealed();
        let (_, recv) = keys();
        let (same_direction, _) = keys();
        assert!(read_all(&wire, recv).is_err(), "the other direction's key refuses them");
        assert_eq!(read_all(&wire, same_direction).unwrap(), vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0]]);
    }

    #[test]
    fn altered_replayed_or_reordered_frames_are_refused() {
        let (wire, first) = sealed();
        let fresh = || keys().0;

        // A payload byte flipped.
        let mut altered = wire.clone();
        altered[first - TAG - 1] ^= 1;
        let err = read_all(&altered, fresh()).unwrap_err();
        assert_eq!(err.to_string(), "frame failed authentication");

        // A dimension changed together with the byte length: a consistent
        // header, refused by its tag before the payload is allocated.
        let mut resized = wire.clone();
        resized[6..14].copy_from_slice(&(1u64 << 30).to_le_bytes());
        resized[14..22].copy_from_slice(&(4u64 << 30).to_le_bytes());
        let err = read_all(&resized, fresh()).unwrap_err();
        assert_eq!(err.to_string(), "frame failed authentication");

        // The first frame sent twice: the copy carries the wrong number.
        let mut replayed = wire[..first].to_vec();
        replayed.extend_from_slice(&wire[..first]);
        assert!(read_all(&replayed, fresh()).is_err());

        // The second frame before the first.
        let mut reordered = wire[first..].to_vec();
        reordered.extend_from_slice(&wire[..first]);
        assert!(read_all(&reordered, fresh()).is_err());
    }

    #[test]
    fn a_short_secret_is_refused_and_the_key_never_prints() {
        assert!(GroupKey::from_secret(b"too short").is_err());
        let key = GroupKey::from_secret(&[7u8; 32]).unwrap();
        assert_eq!(format!("{key:?}"), "GroupKey(..)");
    }
}
