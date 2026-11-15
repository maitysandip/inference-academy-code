"""Mixed precision in numbers: what fp16 and bf16 can and cannot represent, and why loss scaling exists.

fp32: 8 exponent bits, 23 mantissa bits.  fp16: 5 exponent, 10 mantissa.  bf16: 8 exponent, 7 mantissa.
fp16 has more precision but a tiny range (max 65504, smallest normal 6.1e-5); bf16 has fp32's range and
little precision. We emulate both with the standard library: struct's 'e' format is IEEE fp16, and bf16 is
fp32 with the low 16 bits rounded away.
"""
import struct


def fp16(x):
    try:
        return struct.unpack("e", struct.pack("e", x))[0]
    except OverflowError:
        return float("inf")


def bf16(x):
    b = struct.unpack("I", struct.pack("f", x))[0]
    b = ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) << 16  # round to nearest even
    return struct.unpack("f", struct.pack("I", b & 0xFFFFFFFF))[0]


if __name__ == "__main__":
    print(f"{'value':>12} {'fp16':>12} {'bf16':>12}")
    for v in (1.0001, 3.14159265, 1000.7, 70000.0, 1e-5, 1e-8, 1e-30):
        print(f"{v:>12.6g} {fp16(v):>12.6g} {bf16(v):>12.6g}")

    print("\nsmall gradients in fp16, with and without loss scaling (scale 1024):")
    grads = [3e-8, 2e-7, 5e-6, 1e-4]
    for g in grads:
        plain = fp16(g)
        scaled = fp16(g * 1024) / 1024
        print(f"  grad {g:.0e}: fp16 {plain:.3e}   scaled then unscaled {scaled:.3e}")

    print("\nadding a small update to a weight (why optimizers keep fp32 master weights):")
    w, upd = 1.0, 1e-4
    w16, wb, w32 = 1.0, 1.0, 1.0
    for _ in range(1000):
        w16, wb, w32 = fp16(w16 + upd), bf16(wb + upd), w32 + upd
    print(f"  1000 updates of +1e-4 to 1.0: fp32 {w32:.4f}   fp16 {w16:.4f}   bf16 {wb:.4f}")

    print("\nbytes per parameter for mixed-precision Adam training:")
    for name, b in (("bf16 weights", 2), ("bf16 gradients", 2), ("fp32 master weights", 4), ("fp32 Adam m and v", 8)):
        print(f"  {name:<22}{b:>3}")
    print(f"  {'total':<22}{16:>3}  -> a 1B-parameter model needs ~16 GB before activations")
