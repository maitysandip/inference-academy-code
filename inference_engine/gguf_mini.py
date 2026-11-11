"""Write and read a minimal GGUF-style file (subset: string/uint32 metadata, float32 tensors). Follows the published GGUF v3 layout."""
import mmap, os, struct

MAGIC, VERSION, ALIGN = b"GGUF", 3, 32
T_UINT32, T_STRING = 4, 8            # metadata value types
GGML_F32 = 0                          # tensor type id

def w_str(f, s): b = s.encode(); f.write(struct.pack("<Q", len(b))); f.write(b)
def r_str(buf, off):
    (n,) = struct.unpack_from("<Q", buf, off); off += 8
    return bytes(buf[off:off + n]).decode(), off + n

def write_gguf(path, metadata, tensors):
    """metadata: {key: int|str}; tensors: {name: (dims, flat list of floats)}"""
    with open(path, "wb") as f:
        f.write(MAGIC); f.write(struct.pack("<IQQ", VERSION, len(tensors), len(metadata)))
        for k, v in metadata.items():
            w_str(f, k)
            if isinstance(v, int): f.write(struct.pack("<II", T_UINT32, v))
            else: f.write(struct.pack("<I", T_STRING)); w_str(f, v)
        offset = 0; infos = []
        for name, (dims, data) in tensors.items():
            w_str(f, name); f.write(struct.pack("<I", len(dims)))
            for d in dims: f.write(struct.pack("<Q", d))
            f.write(struct.pack("<IQ", GGML_F32, offset))
            n = 4 * len(data); offset += (n + ALIGN - 1) // ALIGN * ALIGN
            infos.append(data)
        pad = (-f.tell()) % ALIGN; f.write(b"\0" * pad)          # tensor data starts at an aligned offset
        for data in infos:
            raw = struct.pack(f"<{len(data)}f", *data); f.write(raw); f.write(b"\0" * ((-len(raw)) % ALIGN))

def read_gguf(path):
    f = open(path, "rb"); mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)    # map the file; no copy yet
    assert mm[:4] == MAGIC, "not a GGUF file"
    version, n_tensors, n_kv = struct.unpack_from("<IQQ", mm, 4); off = 24
    meta = {}
    for _ in range(n_kv):
        k, off = r_str(mm, off); (t,) = struct.unpack_from("<I", mm, off); off += 4
        if t == T_UINT32: (meta[k],) = struct.unpack_from("<I", mm, off); off += 4
        else: meta[k], off = r_str(mm, off)
    infos = []
    for _ in range(n_tensors):
        name, off = r_str(mm, off); (nd,) = struct.unpack_from("<I", mm, off); off += 4
        dims = struct.unpack_from(f"<{nd}Q", mm, off); off += 8 * nd
        ttype, toff = struct.unpack_from("<IQ", mm, off); off += 12
        infos.append((name, dims, ttype, toff))
    data_start = (off + ALIGN - 1) // ALIGN * ALIGN
    def tensor(name):
        for n, dims, tt, toff in infos:
            if n == name:
                count = 1
                for d in dims: count *= d
                return dims, list(struct.unpack_from(f"<{count}f", mm, data_start + toff))
        raise KeyError(name)
    return version, meta, [i[0] for i in infos], tensor

if __name__ == "__main__":
    path = "mini.gguf"
    write_gguf(path, {"general.architecture": "tiny", "tiny.block_count": 2}, {"tok_emb": ((3, 2), [1, 2, 3, 4, 5, 6]), "bias": ((2,), [0.5, -0.5])})
    version, meta, names, tensor = read_gguf(path)
    print(version, meta, names, os.path.getsize(path), "bytes")
    print(tensor("tok_emb"), tensor("bias"))
