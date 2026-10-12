# Filter + compress benchmark: `u32x4`

Date: 2026-10-12

Filter `src[i] > threshold` over a `[]u32` and pack the values that pass into
`dst`, 4 lanes at a time. Two versions:

- **llvm**: `simd.masked_compress_store`. NEON has no compress instruction, so
  LLVM lowers it to one branch per lane (`tbz`/`tbnz` + single-lane `st1`).
- **table**: `neon.movemask` → `SHUFFLE_32x4[mask]` → one `tbl`
  (`simd.runtime_swizzle`) → full 16-byte store → `count += count_ones(mask)`.
  No branches.

## Setup

| | |
|---|---|
| Machine | Apple M3, 16 GB, macOS 26.6.2 |
| Compiler | Odin dev-2026-10 (84bc3fc21), `-o:speed`, bounds checks on |
| Data | 16,777,216 random `u32` (64 MB), `rand.reset(42)` |
| Timing | fastest of 7 runs per cell |
| Check | both versions produce identical output for every row |

## Results

| kept | count | llvm | table | speedup |
|---:|---:|---:|---:|---:|
| 1% | 167,863 | 4.57 ms | 2.86 ms | 1.60× |
| 25% | 4,193,627 | 29.16 ms | 2.94 ms | 9.91× |
| 50% | 8,386,373 | 46.80 ms | 2.99 ms | 15.68× |
| 75% | 12,579,680 | 29.18 ms | 3.04 ms | 9.59× |
| 99% | 16,608,783 | 5.83 ms | 3.14 ms | 1.86× |

A second run was within ±3% on every cell.

### vs a scalar baseline

Added later: a plain loop with no SIMD and no branches (always write
`dst[count] = x`, then `count += int(x > threshold)`), against the same
`table` version. This was a separate quick test, not the harness above: same
data, `-o:speed -no-bounds-check`, fastest of 7, three selectivities, outputs
matched.

| kept | scalar | table | vs scalar |
|---:|---:|---:|---:|
| 1% | 4.18 ms | 2.70 ms | 1.55× |
| 50% | 4.21 ms | 2.85 ms | 1.48× |
| 99% | 4.38 ms | 3.01 ms | 1.45× |

## Takeaways

- **llvm is worst at 50%**: each per-lane branch is a coin flip, so the branch
  predictor misses constantly. At 1% / 99% the branches are predictable and it
  gets much faster.
- **table is flat at ~3 ms** (~21 GB/s) whatever the selectivity: no branches,
  same work every iteration. At 64 MB it's limited by reading memory, not by the
  compress.
- **table still wins in llvm's best case** (1.6–1.9× at 1% / 99%).
- **Against the scalar baseline the win is ~1.5×**: scalar has no branches
  either, and with 4-byte values it moves 4 bytes per step, so both versions
  get close to the memory read speed. (On `u8` the gap is ~3.3×, see
  [benchmark_u8x16.md](benchmark_u8x16.md).)
