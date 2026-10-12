package neon

import "core:simd"

// Moves the lanes whose movemask bit is set to the front of the vector, in
// order. The lanes after them are zero (the table's 42 padding).
//
// Store the whole result, then advance the output by the number of kept lanes
// (`card` / popcount of the movemask): the next store overwrites the zeros.

@(private="file")
shuffle_32x4 :: proc "contextless" (movemask: u8, cols: #simd[4]u32) -> #simd[4]u32 {
	bytes := transmute(#simd[16]u8)cols
	return transmute(#simd[4]u32)simd.runtime_swizzle(bytes, SHUFFLE_32x4[movemask])
}

@(private="file")
shuffle_16x8 :: proc "contextless" (movemask: u8, cols: #simd[8]u16) -> #simd[8]u16 {
	bytes := transmute(#simd[16]u8)cols
	return transmute(#simd[8]u16)simd.runtime_swizzle(bytes, SHUFFLE_16x8[movemask])
}

// 16 lanes are done one 8-lane half at a time, so there are two results:
// `lo` holds the kept lanes from 0-7 and `hi` the kept lanes from 8-15. Store
// `lo`, advance by the kept count of the low byte of the movemask, then store
// `hi` and advance by the kept count of the high byte.
@(private="file")
shuffle_8x16 :: proc "contextless" (movemask: u16, cols: #simd[16]u8) -> (lo, hi: #simd[16]u8) {
	lo = simd.runtime_swizzle(cols, SHUFFLE_8x16[movemask & 0xFF])
	// The table points into the low half; +8 points it at the high half.
	hi = simd.runtime_swizzle(cols, SHUFFLE_8x16[movemask >> 8] + 8)
	return
}

shuffle :: proc{shuffle_32x4, shuffle_16x8, shuffle_8x16}
