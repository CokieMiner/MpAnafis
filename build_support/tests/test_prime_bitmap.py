"""Exhaustive odd-integer lookup checks against an independent full-integer sieve."""

from math import isqrt

from support import BuildScriptCase


class PrimeBitmapTests(BuildScriptCase):
    def test_lookup_bits_match_an_independent_sieve_and_target_domain(self):
        limit = 1 << 20
        primes = bytearray([1]) * limit
        primes[0:2] = b"\x00\x00"
        for prime in range(2, isqrt(limit - 1) + 1):
            if primes[prime]:
                start = prime * prime
                count = (limit - 1 - start) // prime + 1
                primes[start:limit:prime] = bytes(count)
        for width, byte_count in (("16", 1 << 12), ("32", 1 << 16), ("64", 1 << 16)):
            with self.subTest(pointer_width=width):
                self.run_build(CARGO_CFG_TARGET_POINTER_WIDTH=width)
                bitmap = self.bitmap.read_bytes()
                self.assertEqual(len(bitmap), byte_count)
                for number in range(1, byte_count * 16, 2):
                    composite = bool(bitmap[number >> 4] & (1 << ((number >> 1) & 7)))
                    if composite != (not primes[number]):
                        self.fail(f"incorrect lookup bit for {number} on {width}-bit target")
