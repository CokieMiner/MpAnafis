import unittest
from asm_analyzer.features.stlf import analyze_stlf_hazards


class TestStlfPredictor(unittest.TestCase):
    def test_clean_independent_memory(self):
        asm = """
        movq (%rsi), %rax
        movq (%rdi), %rdx
        movq %rax, (%rcx)
        """
        stats = analyze_stlf_hazards(asm)
        self.assertFalse(stats.has_stlf_hazard)

    def test_stlf_offset_mismatch_hazard(self):
        asm = """
        movq %rax, 0(%rsi)
        movq 4(%rsi), %rdx
        """
        stats = analyze_stlf_hazards(asm)
        self.assertTrue(stats.has_stlf_hazard)
        self.assertGreater(stats.max_penalty_cycles, 0.0)


if __name__ == "__main__":
    unittest.main()
