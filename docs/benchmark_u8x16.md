# Filter + compress benchmark: `u8x16`

Date: 2026-10-12

Same benchmark as [benchmark_u32x4.md](benchmark_u32x4.md), on bytes: filter
`src[i] > threshold` over a `[]u8` and pack the values that pass into `dst`,
16 lanes at a time. Three versions:

- **llvm**: `simd.masked_compress_store`, lowered to one branch per lane, so
  16 branches per iteration.
- **table**: `neon.movemask` (16-bit) → `neon.shuffle`, which looks up
  `SHUFFLE_8x16` once per 8-lane half (`+ 8` on the indices for the high half)
  → two `tbl`s → two 16-byte stores, each followed by
  `count += count_ones(half of mask)`. No branches.
- **scalar**: a plain loop with no SIMD and no branches: always write
  `dst[count] = x`, then `count += int(x > threshold)`. The fair baseline.

## Setup

| | |
|---|---|
| Machine | Apple M3, 16 GB, macOS 26.6.2 |
| Compiler | Odin dev-2026-10 (84bc3fc21), `-o:speed`, with and without `-no-bounds-check` |
| Data | 67,108,864 random `u8` (64 MB, same bytes as the `u32x4` run), `rand.reset(42)` |
| Timing | fastest of 7 runs per cell |
| Check | all versions produce identical output for every row |

`u8` only has 256 values, so the kept fractions land near the targets (1.2%
instead of 1%, 98.8% instead of 99%).

## Results

Bounds checks off (`-no-bounds-check`):

| kept | count | llvm | scalar | table | vs llvm | vs scalar |
|---:|---:|---:|---:|---:|---:|---:|
| 1.2% | 787,916 | 15.13 ms | 16.79 ms | 5.28 ms | 2.87× | 3.18× |
| 25.0% | 16,777,441 | 91.63 ms | 17.35 ms | 5.28 ms | 17.37× | 3.29× |
| 50.0% | 33,550,596 | 140.80 ms | 17.42 ms | 5.34 ms | 26.35× | 3.26× |
| 75.0% | 50,330,541 | 91.21 ms | 17.71 ms | 5.38 ms | 16.95× | 3.29× |
| 98.8% | 66,321,742 | 17.68 ms | 17.78 ms | 5.46 ms | 3.24× | 3.26× |

Bounds checks on:

| kept | count | llvm | scalar | table | vs llvm | vs scalar |
|---:|---:|---:|---:|---:|---:|---:|
| 1.2% | 787,916 | 15.56 ms | 19.48 ms | 5.72 ms | 2.72× | 3.41× |
| 25.0% | 16,777,441 | 91.48 ms | 19.21 ms | 5.48 ms | 16.71× | 3.51× |
| 50.0% | 33,550,596 | 140.81 ms | 19.13 ms | 5.72 ms | 24.64× | 3.35× |
| 75.0% | 50,330,541 | 91.52 ms | 19.28 ms | 5.79 ms | 15.80× | 3.33× |
| 98.8% | 66,321,742 | 17.07 ms | 19.53 ms | 5.95 ms | 2.87× | 3.28× |

An earlier two-version run (llvm vs table only) gave the same llvm and table
numbers within ±5%.

## Takeaways

- **Against the fair baseline, table is ~3.3× faster** at every selectivity.
  scalar handles one byte per step and each step waits for the previous
  `count` (~1 byte per cycle); table does 16 bytes per step.
- **Most of the huge "vs llvm" numbers come from llvm's weak fallback**: the
  plain scalar loop already beats `masked_compress_store` at 25–75%, because it
  has no branches to mispredict.
- **llvm has 16 data-dependent branches per iteration** here (vs 4 for
  `u32x4`), which is why it's even worse on bytes.
- **table is flat** (~5.3–5.7 ms) across selectivities, but about 2× slower
  than the `u32x4` table run on the same 64 MB: per 16 bytes it does two
  lookups, two `tbl`s and two stores, and the second store's address depends on
  the first popcount.
- **Bounds checks cost scalar ~10%** (one check per value) and table very
  little (two checks per 16 bytes).
