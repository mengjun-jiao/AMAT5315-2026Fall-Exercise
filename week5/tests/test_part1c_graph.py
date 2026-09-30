"""Automated validation for Week 5 Part 1C JAXPR graph rendering."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

import part1c_graph


def test_recorded_operation_sequences() -> None:
    primal_labels = part1c_graph.operation_sequence(part1c_graph.primal_jaxpr())
    gradient_labels = part1c_graph.operation_sequence(part1c_graph.gradient_jaxpr())
    assert primal_labels == part1c_graph.PRIMAL_OPERATION_SEQUENCE
    assert "add_any" in gradient_labels


def test_gradient_shared_node_accumulation() -> None:
    gradient_jaxpr = part1c_graph.gradient_jaxpr()
    gradient_labels = part1c_graph.operation_sequence(gradient_jaxpr)
    accumulation = part1c_graph.shared_node_accumulation(gradient_jaxpr)
    contribution_equations = accumulation["contribution_equations"]
    assert gradient_labels[accumulation["add_index"]] == "add_any"
    assert [equation.primitive.name for _, equation in contribution_equations] == [
        "neg",
        "mul",
    ]

    producers = accumulation["producer_map"]
    neg_index, neg_equation = contribution_equations[0]
    mul_index, mul_equation = contribution_equations[1]
    assert neg_index < accumulation["add_index"]
    assert mul_index < accumulation["add_index"]

    neg_source_index, neg_source = producers[id(neg_equation.invars[0])]
    assert neg_source_index < neg_index
    assert neg_source.primitive.name == "mul"
    assert any(hasattr(variable, "val") and float(variable.val) == 4.0 for variable in neg_source.invars)

    mul_sources = [producers[id(variable)] for variable in mul_equation.invars]
    assert sorted(equation.primitive.name for _, equation in mul_sources) == ["mul", "mul"]
    derivative_scale = [
        (index, equation)
        for index, equation in mul_sources
        if any(hasattr(variable, "val") and float(variable.val) == 2.0 for variable in equation.invars)
    ]
    assert len(derivative_scale) == 1
    derivative_index, derivative_equation = derivative_scale[0]
    assert derivative_index < mul_index
    derivative_input_index, derivative_input = producers[id(derivative_equation.invars[-1])]
    assert derivative_input_index < derivative_index
    assert derivative_input.primitive.name == "integer_pow"
    assert derivative_input.params["y"] == 1

    power_two = next(
        equation
        for equation in gradient_jaxpr.eqns
        if equation.primitive.name == "integer_pow" and equation.params["y"] == 2
    )
    power_two_input_index, power_two_input = producers[id(power_two.invars[0])]
    assert power_two_input_index < derivative_index
    assert power_two_input.primitive.name == "integer_pow"
    assert power_two_input.params["y"] == -6

    multiplication_literals = [
        float(variable.val)
        for equation in gradient_jaxpr.eqns
        if equation.primitive.name == "mul"
        for variable in equation.invars
        if hasattr(variable, "val")
    ]
    assert -6.0 in multiplication_literals
    assert 2.0 in multiplication_literals
    assert 4.0 in multiplication_literals

    sub_equation = next(
        equation for equation in gradient_jaxpr.eqns if equation.primitive.name == "sub"
    )
    c_output = sub_equation.outvars[0]
    primal_energy_equation = next(
        equation
        for equation in gradient_jaxpr.eqns
        if equation.primitive.name == "mul"
        and any(id(variable) == id(c_output) for variable in equation.invars)
        and any(hasattr(variable, "val") and float(variable.val) == 4.0 for variable in equation.invars)
    )
    assert primal_energy_equation is not None

    forward_labels = part1c_graph.operation_sequence(part1c_graph.primal_jaxpr())
    assert forward_labels == part1c_graph.PRIMAL_OPERATION_SEQUENCE
    assert "sub" in gradient_labels
    assert "integer_pow[y=-7]" in gradient_labels
    assert "integer_pow[y=2]" in gradient_labels
    assert "mul" in gradient_labels


def test_graph_images_are_generated() -> None:
    graph_path, gradient_path = part1c_graph.render_graphs(Path("artifacts/ad"))
    assert graph_path.is_file()
    assert gradient_path.is_file()
    assert graph_path.stat().st_size > 0
    assert gradient_path.stat().st_size > 0


if __name__ == "__main__":
    test_recorded_operation_sequences()
    test_gradient_shared_node_accumulation()
    test_graph_images_are_generated()
    print("Part 1C tests passed")
