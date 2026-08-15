import unittest
from asm_analyzer.features.short_loop import analyze_short_loops


class TestShortLoopPredictor(unittest.TestCase):
    def test_basic_short_loop_prediction(self):
        asm = """
        mov %rdx, %rcx
        test %rcx, %rcx
        jz 2f
        .p2align 4
        1:
        movq (%rsi), %rax
        adcq (%rdi), %rax
        movq %rax, (%rdx)
        dec %rcx
        jnz 1b
        2:
        ret
        """
        stats = analyze_short_loops(asm, loop_body_cycles_per_iter=1.0, unroll_factor=1)
        self.assertIn("estimated_cycles_by_limbs", stats)
        predictions = stats["estimated_cycles_by_limbs"]
        self.assertIn(1, predictions)
        self.assertIn(2, predictions)
        self.assertIn(4, predictions)
        self.assertIn(8, predictions)
        # N=2 cost should be higher than N=1, but sublinear due to fixed mispredict overhead
        self.assertGreater(predictions[2], predictions[1])
        self.assertGreater(predictions[4], predictions[2])


if __name__ == "__main__":
    unittest.main()
