# Optimized Stream Compression w/ Odin

When writing filtering kernels with SIMD, the comparison instruction (`vcgtq`) returns masks. NEON (SIMD extension for ARM platforms) has no instructions to apply masks to vectors so that they only contain values that satisfy the predicate. To perform the vector compression in Odin, one is typically gonna use `simd.masked_compress_store`.

This is fine for AVX-512 platforms that have a vector compression instruction. With NEON, LLVM actually falls back to scalar code! This is annoying because the physical execution is made faster with SIMD and you lose all the benefit when the results need to be collected!

This repository contains my attempt at emulating the `VCOMPRESSD` instruction on NEON using a lookup table. This technique has existed for many years in database engines and honestly, I wasn't able to retrace who created it. However, I found [this repo (`lemire/simdprune`)](https://github.com/lemire/simdprune) which implements the technique! It was written by [Daniel Lemire](https://github.com/lemire), the guy that made `simdjson` among many other amazing things. This guy is really impressive!

## How it works

Imagine you just wrote the comparison loop for 32-bit unsigned integers and you get a mask like that for every iteration:

```c
(0xFFFFFFFF, 0x0, 0x0, 0xFFFFFFFF)
```

You could basically use the metaphor of border security at the airport (give me a visa please):

```c
// Y = Yes
// N = No
(Y, N, N, Y)
```

To be honest, masks are a great idea but applying them proved pretty hard for me. The basic version looks like this (pseudo-code):

```go
// an array of results from the query, I know that representing age as u32 is
// wasteful. I was just tired and I did not want to write 16 if-statements for
// the 16 simd vector lanes.
result := *u32{}

threshold := vdupq(18)
numbers   := vld1q(age_col[:i])
can_drink := vcgtq(numbers, threshold)

print(can_drink)
// (0xFFFFFFFF, 0x0, 0x0, 0xFFFFFFFF)

if (can_drink[0] != false) { append(result, &age_col[i + 0]) }
if (can_drink[1] != false) { append(result, &age_col[i + 1]) }
if (can_drink[2] != false) { append(result, &age_col[i + 2]) }
if (can_drink[3] != false) { append(result, &age_col[i + 3]) }
```

As you can see, this branches a lot... This is not very nice for the CPU! To do this faster, we're gonna use a LUT (lookup table). In this project, because it was so painful to write by hand and I would have made mistakes, I [wrote a python script](utils/gen_tables.py) to do it for me.

We basically take a mask and represent it as a number. It's just an array of boolean so it's easy enough to bitpack:

```c
(0xFFFFFFFF, 0x0, 0x0, 0x0) -> 0b0001   // lane 0 kept = rightmost bit
```

Because we have four lanes, that means we have 16 possible permutations from `0b0000` to `0b1111`! This is just an array of 16 entries (16 bytes each) that can be represented at compile-time.

```odin
// 4 byte(s) per lane; 42 = unused output byte and the answer to anything
@(rodata)
SHUFFLE_32x4 := [16]#simd[16]u8{
	0b0000 = {42, 42, 42, 42,  42, 42, 42, 42,  42, 42, 42, 42,  42, 42, 42, 42},
	0b0001 = { 0,  1,  2,  3,  42, 42, 42, 42,  42, 42, 42, 42,  42, 42, 42, 42},
	0b0010 = { 4,  5,  6,  7,  42, 42, 42, 42,  42, 42, 42, 42,  42, 42, 42, 42},
	0b0011 = { 0,  1,  2,  3,   4,  5,  6,  7,  42, 42, 42, 42,  42, 42, 42, 42},
	// ... truncated for sanity and brevity
```

Okay so you might be screaming right now "Why does it tell me which bytes to select from the array and not the lane index?". It's because of another NEON quirk, it doesn't have a shuffle instruction that takes lane indices at runtime so we have to tell it what to select using `tbl`.

Here's an example of why TBL is useful!

```c
// WADE = walk with effort through water or another liquid or viscous substance
secret_word = [W A D E]
indices = [0, 3, 3, 2]

tbl(secret_word, indices)
// WEED
```

Let's see how we can apply it to the LUT:

```odin
// cols     = the 4 numbers we filtered, e.g. (10, 20, 30, 40)
// movemask = the mask as a number, e.g. 0b1010 (lanes 1 and 3 passed)
//
// 1. We take the cols passed by the filtering loop and interpret them as a simple byte array.
bytes := transmute(#simd[16]u8)cols
// We get the precomputed layout we want, for the mask we have
shuffle_table := SHUFFLE_32x4[movemask]
// And then use it to extract the bytes we want from the array
junk := simd.runtime_swizzle(bytes, shuffle_table)
// We reinterpret the junk we got as a vector of 4 u32s
// This yields the numbers that have passed the filter, contiguous in memory!
numbers := transmute(#simd[4]u32)junk
```

Then we can take the vector and pack it inside a buffer, which was also hard for me to do:

```odin
m := neon.movemask(simd.lanes_gt(v, pred))   // e.g. 0b1010
numbers := neon.shuffle(m, v)                 // e.g. (20, 40, 0, 0)
// Write all 4 lanes, zeros included
intrinsics.unaligned_store((^simd.u32x4)(&dst[count]), numbers)
// Move forward only by how many passed: the next store overwrites the zeros
count += int(intrinsics.count_ones(m))
```

That's pretty much all! There's another optimization and it's pretty minute and out of scope for the stream compression. There is already a lot of ground covered here so for tonight, I'm gonna rest.

## Benchmarks

Filtering 64 MB of random junk (`/dev/urandom`) on an Apple M3 (fastest of 7 runs). The full numbers are in [`docs/benchmark_u32x4.md`](docs/benchmark_u32x4.md) and [`docs/benchmark_u8x16.md`](docs/benchmark_u8x16.md).

| | `masked_compress_store` | plain scalar loop | lookup table | vs `masked_compress_store` | vs scalar |
|---|---:|---:|---:|---:|---:|
| `u32`, 50% kept | 46.80 ms | 4.21 ms | 2.99 ms | 15.7× | 1.4× |
| `u8`, 50% kept | 140.80 ms | 17.42 ms | 5.34 ms | 26.4× | 3.3× |

## DISCLAIMER

I am pretty happy I finished this because it broke a lot of barriers for me. I was not familiar with SIMD and I only discovered it recently thanks to [Hashimoto's blog post on the subject](https://mitchellh.com/writing/everyone-should-know-simd). It feels like magic honestly, the speed-ups are unreal. The code is gnarly and interesting to write, some of the APIs do feel a little esoteric and it's hard to think about portability.

This was quite heavily AI-assisted. I know, I know! I don't want to make slop too but I'm just a 19-year-old guy that's not very good at mathematics or low-level programming. I can have an intuition of what needs to be done but all that arithmetic formalization is still too hard for me.

Especially the "zeroing out the high bits" with bitwise ops like `byte & 0xFF`. Because no one can teach me, I am resorting to our common artificial god (lol) to teach me stuff and be my friend for the night.

If you, a real human of bones and flesh, of spirit and bravado want to be my friend and go on a venture together, please shoot me an email at `noctisatrae [at] pm.me`.

Love!

## Looking forward

I always dreamt of being a better open-source contributor. I would love to be able to contribute this upstream to Odin. Even crazier, to the LLVM compiler which would probably be an arduous and complex project. As an outsider to the world of high-quality widely-used open-source projects, I don't know what the process to work on something like that is!
