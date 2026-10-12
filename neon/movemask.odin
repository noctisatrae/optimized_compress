package neon

import "core:simd"

@(private="file")
movemask_8x16 :: proc "contextless" (mask: #simd[16]u8) -> u16 {
	// A u8 lane can't hold 256 or more, so do the two halves of 8 lanes
	// separately and put the high half in the top byte.
	BITS :: #simd[16]u8{1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128}
	m := mask & BITS
	lo := simd.swizzle(m, 0, 1, 2, 3, 4, 5, 6, 7)
	hi := simd.swizzle(m, 8, 9, 10, 11, 12, 13, 14, 15)
	return u16(simd.reduce_or(lo)) | u16(simd.reduce_or(hi)) << 8
}

@(private="file")
movemask_16x8 :: proc "contextless" (mask: #simd[8]u16) -> u8 {
	BITS :: #simd[8]u16{1, 2, 4, 8, 16, 32, 64, 128}
	return u8(simd.reduce_or(mask & BITS))
}

@(private="file")
movemask_32x4 :: proc "contextless" (mask: #simd[4]u32) -> u8 {
	BITS :: #simd[4]u32{1, 2, 4, 8}
	return u8(simd.reduce_or(mask & BITS))
}

movemask :: proc{movemask_8x16, movemask_16x8, movemask_32x4}
