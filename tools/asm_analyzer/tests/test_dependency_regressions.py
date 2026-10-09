"""Dependency proofs and bounded scheduling regressions."""

import itertools
import random
import unittest

from asm_analyzer.search.adapters import adapter_for
from asm_analyzer.search.ast import get_instruction_spec, parse_line, parse_operand
from asm_analyzer.search.dag import (
    DagNode,
    build_dag,
    evaluate_schedule_cost,
    exact_optimal_schedule,
    relax_anti_dependencies_with_renaming,
    topological_permutations,
    simulated_annealing_permutations,
)
from asm_analyzer.search.ast import Instr, Spec
from asm_analyzer.search.engine import _schedulable_segments
from asm_analyzer.search.memory_dependencies import may_alias
from asm_analyzer.types import ArchitectureFamily


class TestDependencyRegressions(unittest.TestCase):
    def test_random_dag_schedules_preserve_every_edge(self):
        for seed in range(20):
            random_source = random.Random(seed)
            size = 16
            edges = {(left, right) for left in range(size) for right in range(left + 1, size) if random_source.random() < 0.2}
            nodes = [DagNode(index, Instr("nop", "nop", []), Spec(),
                             {left for left, right in edges if right == index},
                             {right for left, right in edges if left == index}) for index in range(size)]
            for order in simulated_annealing_permutations(nodes, count=12, seed=seed):
                self.assertEqual(sorted(order), list(range(size)))
                positions = {node: index for index, node in enumerate(order)}
                self.assertTrue(all(positions[left] < positions[right] for left, right in edges))

    def test_mips_delay_slot_is_fixed(self):
        adapter = adapter_for(ArchitectureFamily.MIPS64)
        lines = ["daddu $4, $5, $6", "bnez $4, 1f", "daddu $7, $8, $9", "daddu $10, $11, $12", "1:"]
        segments, error = _schedulable_segments(lines, adapter.parse, delay_slots=True)
        self.assertEqual(error, "")
        positions = {position for indices, _ in segments for position in indices}
        self.assertEqual(positions, {0, 3})
        _, error = _schedulable_segments(lines[:2], adapter.parse, delay_slots=True)
        self.assertIn("delay slot", error)

    def test_three_operand_inputs_remain_ordered(self):
        for operation in ("imulq $3, %rcx, %rax", "shldq $3, %rcx, %rax"):
            with self.subTest(operation=operation):
                nodes = build_dag([parse_line("movq %r8, %rcx"), parse_line(operation)])
                self.assertIn(0, nodes[1].preds)

    def test_partial_flag_writes_preserve_full_flag_readers(self):
        nodes = build_dag(list(map(parse_line, (
            "seto %al", "adcxq %r8, %r9", "adoxq %r10, %r11",
        ))))
        self.assertIn(0, nodes[1].preds)
        self.assertIn(0, nodes[2].preds)

    def test_stack_operations_are_barriers_with_implicit_rsp_effects(self):
        for instruction in ("pushq %rax", "popq %rax"):
            spec = get_instruction_spec(parse_line(instruction))
            self.assertIn("rsp", spec.uses)
            self.assertIn("rsp", spec.defs)
            self.assertTrue(spec.unknown)

    def test_partial_destinations_read_the_prior_full_register(self):
        for instruction in ("movb $1, %al", "movw $1, %ax", "setc %al"):
            self.assertIn("rax", get_instruction_spec(parse_line(instruction)).uses)

    def test_index_only_addresses_retain_scale_and_position(self):
        operand = parse_operand("08(,%rcx,8)")
        self.assertIsNone(operand.base)
        self.assertEqual((operand.index, operand.scale, operand.displacement), ("rcx", 8, 8))
        self.assertTrue(may_alias(operand, parse_operand("8(%rcx)")))

    def test_symbolic_addresses_never_prove_disjoint(self):
        for address in ("symbol(%rdi)", "%fs:8(%rax)", "symbol+8(%rip)"):
            operand = parse_operand(address)
            self.assertEqual(operand.kind, "mem")
            self.assertIsNone(operand.addr)
            self.assertTrue(may_alias(operand, parse_operand("64(%rdi)")))

    def test_assembler_specific_leading_zero_radix_cannot_prove_disjoint(self):
        self.assertIsNone(parse_operand("010(%rdi)").addr)

    def test_unknown_banks_and_memory_exchange_are_barriers(self):
        for instruction in ("movq %xmm0, %rax", "xchgq (%rdi), %rax", "movq symbol, %rax"):
            self.assertTrue(get_instruction_spec(parse_line(instruction)).unknown)

    def test_assembly_comments_do_not_hide_register_dependencies(self):
        instruction = parse_line("movq %rax, %rbx # explanatory text")
        self.assertEqual(get_instruction_spec(instruction).defs, {"rbx"})

    def test_native_access_widths_preserve_overlapping_ranges(self):
        for arch, store, load, width in (
            (ArchitectureFamily.AARCH64, "str x0, [x1]", "ldr w2, [x1, #4]", 64),
            (ArchitectureFamily.AARCH64, "stp x0, x2, [x1]", "ldr x3, [x1, #8]", 128),
            (ArchitectureFamily.ARM32, "str r0, [r1]", "ldrb r2, [r1, #2]", 32),
            (ArchitectureFamily.POWER64, "std 3, 0(4)", "lwz 5, 4(4)", 64),
        ):
            with self.subTest(arch=arch, store=store):
                adapter = adapter_for(arch)
                instructions = [adapter.parse(store), adapter.parse(load)]
                nodes = build_dag(instructions, spec_provider=adapter.spec)
                memory = next(op for op in instructions[0].ops if op.kind == "mem")
                self.assertEqual(memory.width, width)
                self.assertIn(0, nodes[1].preds)

    def test_native_indexed_addresses_remain_unknown_aliases(self):
        adapter = adapter_for(ArchitectureFamily.AARCH64)
        instructions = [adapter.parse(line) for line in (
            "str x0, [x1, x2]", "ldr x3, [x1, #64]",
        )]
        self.assertIn(0, build_dag(instructions, spec_provider=adapter.spec)[1].preds)

    def test_native_mnemonic_prefixes_do_not_grant_semantics(self):
        adapter = adapter_for(ArchitectureFamily.AARCH64)
        self.assertTrue(adapter.spec(adapter.parse("add_unknown x0, x1, x2")).unknown)
        self.assertTrue(adapter.spec(adapter.parse("stxr w0, x1, [x2]")).unknown)

    def test_no_renaming_without_live_out_and_scratch_contracts(self):
        instructions = list(map(parse_line, (
            "movq $1, %r8", "addq %r8, %rax", "movq $2, %r8", "addq %r8, %rbx",
        )))
        for kwargs in ({}, {"available_spares": ["r10"]},
                       {"available_spares": ["r10"], "live_out": {"r8"}}):
            renamed = relax_anti_dependencies_with_renaming(
                instructions, ArchitectureFamily.X86_64, **kwargs,
            )
            self.assertEqual(renamed, instructions)

    def test_renaming_does_not_move_read_modify_write_to_uninitialized_spare(self):
        instructions = list(map(parse_line, ("movq $1, %r8", "addq $2, %r8", "movq %r8, %rax")))
        self.assertEqual(relax_anti_dependencies_with_renaming(
            instructions, ArchitectureFamily.X86_64, available_spares=["r10"], live_out={"rax"},
        ), instructions)

    def test_exact_search_reports_budget_exhaustion(self):
        nodes = build_dag([parse_line(f"movq $1, %r{index}") for index in range(8, 16)])
        self.assertIsNone(exact_optimal_schedule(nodes, max_states=20))
        self.assertLessEqual(len(topological_permutations(nodes, count=1)), 1)
        self.assertEqual(topological_permutations(nodes, count=0), [])

    def test_exact_solver_matches_exhaustive_oracle(self):
        nodes = build_dag(list(map(parse_line, (
            "movq (%rdi), %rax", "movq 8(%rdi), %rcx", "addq %rax, %r8", "movq %rcx, %r9",
        ))))
        costs = []
        for order in itertools.permutations(range(len(nodes))):
            positions = {node: index for index, node in enumerate(order)}
            if all(positions[pred] < positions[node.idx] for node in nodes for pred in node.preds):
                costs.append(evaluate_schedule_cost(order, nodes))
        self.assertEqual(evaluate_schedule_cost(exact_optimal_schedule(nodes), nodes), min(costs))


if __name__ == "__main__":
    unittest.main()
