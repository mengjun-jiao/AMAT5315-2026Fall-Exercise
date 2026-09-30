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
    gradient_labels = part1c_graph.operation_sequence(part1c_graph.gradient_jaxpr())
    add_index = gradient_labels.index("add_any")
    assert add_index > 0
    assert gradient_labels[add_index - 1] == "mul"
    assert "neg" in gradient_labels[:add_index]


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
