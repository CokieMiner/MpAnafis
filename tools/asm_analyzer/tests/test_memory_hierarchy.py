"""Unit tests for memory hierarchy working-set analyzer."""

from __future__ import annotations

import unittest
from asm_analyzer.features.memory_hierarchy import analyze_memory_hierarchy


class TestMemoryHierarchyAnalysis(unittest.TestCase):
    def test_small_limb_footprint_in_l1d(self) -> None:
        # 64 limbs * 8 bytes * 3 operands = 1,536 bytes (fits in L1D)
        stats = analyze_memory_hierarchy(limb_count=64, num_operands=3)
        self.assertEqual(stats.cache_tier, "L1D")
        self.assertFalse(stats.spills_l1d)
        self.assertFalse(stats.suggest_cache_blocking)

    def test_large_limb_footprint_in_l3_or_dram(self) -> None:
        # 2,000,000 limbs * 8 bytes * 3 operands = 48 MB (spills into DRAM / L3)
        stats = analyze_memory_hierarchy(limb_count=2_000_000, num_operands=3)
        self.assertEqual(stats.cache_tier, "DRAM")
        self.assertTrue(stats.spills_l1d)
        self.assertTrue(stats.spills_l2)
        self.assertTrue(stats.suggest_cache_blocking)


if __name__ == "__main__":
    unittest.main()
