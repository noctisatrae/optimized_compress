// In this file, I am trying to build a fast algorithm for array compressing
// using a mask computed through a SIMD filter kernel. As you probably know,
// NEON doesn't have compress instructions like AVx-512 so it makes it pretty
// hard to actually build an array after the filtering is done.
//
// The first step is to take the mask and turn it into an unisgned integer:
//
// ```
// mask = {0x0, 0xFFFF, 0x0, 0xFFFF}
// umask = 0101
// ```
//
// From that integer, we know what we can add to the record batch array:
//
// ```
// switch umask {
// 		case 0101:
//         append(&batch[count], v[1], v[3])
// 		case 1010:
// 			append(&batch[count], v[0], v[2])
// 		... etc ...
// }
// ```
//
// I don't want to copy the data around, so I wonder if the v[n] can be added to
// the record batch as a pointer.

package main

import "base:intrinsics"
import "core:fmt"
import "core:math/rand"
import "core:simd"
import "core:slice"
import "core:time"

import "neon"

N    :: 64 * 1024 * 1024 // values to filter (64 MB of u8), must be a multiple of 16
RUNS :: 7                // each filter runs this many times; the fastest run counts

// The simd_kernels version: LLVM's masked_compress_store, which on NEON turns
// into one branch per lane (16 per iteration here).
filter_llvm :: proc "contextless" (dst, src: []u8, threshold: u8) -> (count: int) {
	pred := simd.u8x16(threshold)
	for i := 0; i < len(src); i += 16 {
		v := intrinsics.unaligned_load((^simd.u8x16)(&src[i]))
		mask := simd.lanes_gt(v, pred)
		simd.masked_compress_store(&dst[count], v, mask)
		count += int(simd.reduce_add_ordered(mask & simd.u8x16(1)))
	}
	return
}

// Our version: movemask -> two shuffle table lookups (one per 8-lane half) ->
// two `tbl`s, no branches. Each store writes 16 bytes, so `dst` needs 16 spare
// bytes at the end.
filter_table :: proc "contextless" (dst, src: []u8, threshold: u8) -> (count: int) {
	pred := simd.u8x16(threshold)
	for i := 0; i < len(src); i += 16 {
		v := intrinsics.unaligned_load((^simd.u8x16)(&src[i]))
		m := neon.movemask(simd.lanes_gt(v, pred))
		lo, hi := neon.shuffle(m, v)
		intrinsics.unaligned_store((^simd.u8x16)(&dst[count]), lo)
		count += int(intrinsics.count_ones(m & 0xFF))
		intrinsics.unaligned_store((^simd.u8x16)(&dst[count]), hi)
		count += int(intrinsics.count_ones(m >> 8))
	}
	return
}

// The plain-loop baseline: no SIMD and no branches. Every value is written,
// but `count` only moves forward when it passed, so rejected values get
// overwritten by the next one.
filter_scalar :: proc "contextless" (dst, src: []u8, threshold: u8) -> (count: int) {
	for x in src {
		dst[count] = x
		count += int(x > threshold)
	}
	return
}

Filter :: #type proc "contextless" (dst, src: []u8, threshold: u8) -> int

// Runs `filter` RUNS times and returns the fastest time and the kept count.
bench :: proc(filter: Filter, dst, src: []u8, threshold: u8) -> (best: time.Duration, count: int) {
	best = max(time.Duration)
	for _ in 0 ..< RUNS {
		start := time.tick_now()
		count = filter(dst, src, threshold)
		best = min(best, time.tick_since(start))
	}
	return
}

main :: proc() {
	src := make([]u8, N)
	defer delete(src)
	// +16 spare bytes: the last 16-byte store can go past the kept data.
	dst_llvm := make([]u8, N + 16)
	defer delete(dst_llvm)
	dst_table := make([]u8, N + 16)
	defer delete(dst_table)
	dst_scalar := make([]u8, N + 16)
	defer delete(dst_scalar)

	rand.reset(42)
	_ = rand.read(src)

	fmt.printfln("%d values, fastest of %d runs\n", N, RUNS)
	fmt.printfln("%6s  %12s  %10s  %10s  %10s  %11s  %11s", "kept", "count", "llvm", "scalar", "table", "vs llvm", "vs scalar")

	// How selective the filter is changes how predictable LLVM's branches are,
	// so try a few: almost nothing kept, half kept (random branches), almost
	// everything kept.
	for keep in ([]f64{0.01, 0.25, 0.50, 0.75, 0.99}) {
		// u8 only has 256 values, so the real kept fraction is a bit off the
		// target; the table prints the real one.
		threshold := u8(f64(max(u8)) * (1 - keep))

		t_llvm, n_llvm := bench(filter_llvm, dst_llvm, src, threshold)
		t_scalar, n_scalar := bench(filter_scalar, dst_scalar, src, threshold)
		t_table, n_table := bench(filter_table, dst_table, src, threshold)

		// All versions must produce exactly the same output.
		if n_llvm != n_table || !slice.equal(dst_llvm[:n_llvm], dst_table[:n_table]) ||
		   n_llvm != n_scalar || !slice.equal(dst_llvm[:n_llvm], dst_scalar[:n_scalar]) {
			fmt.eprintfln("MISMATCH at keep=%.2f: llvm=%d scalar=%d table=%d", keep, n_llvm, n_scalar, n_table)
			return
		}

		ms_llvm := time.duration_milliseconds(t_llvm)
		ms_scalar := time.duration_milliseconds(t_scalar)
		ms_table := time.duration_milliseconds(t_table)
		// A width on a number pads it with zeros, so format each cell as a
		// string first and pad that (strings pad with spaces).
		fmt.printfln(
			"%6s  %12s  %10s  %10s  %10s  %11s  %11s",
			fmt.tprintf("%.1f%%", f64(n_llvm) / N * 100),
			fmt.tprintf("%d", n_llvm),
			fmt.tprintf("%.2fms", ms_llvm),
			fmt.tprintf("%.2fms", ms_scalar),
			fmt.tprintf("%.2fms", ms_table),
			fmt.tprintf("%.2fx", ms_llvm / ms_table),
			fmt.tprintf("%.2fx", ms_scalar / ms_table),
		)
	}
}
