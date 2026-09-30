"""Render the JAX-recorded primal and gradient computational graphs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

jax.config.update("jax_enable_x64", True)

try:
    from part1a_ad import energy, primal
except ModuleNotFoundError:
    from scripts.part1a_ad import energy, primal


R_VALUE = 1.3
PRIMAL_OPERATION_SEQUENCE = ["integer_pow[y=-6]", "integer_pow[y=2]", "sub", "mul"]


def scalar_input() -> Any:
    return jnp.asarray(R_VALUE, dtype=jnp.float64)


def primal_jaxpr() -> Any:
    """Record the explicit Part 1A primal node sequence with JAX."""
    return jax.make_jaxpr(primal)(scalar_input()).jaxpr


def gradient_jaxpr() -> Any:
    """Record JAX's reverse-mode gradient of the Part 1A scalar energy."""
    return jax.make_jaxpr(jax.grad(energy))(scalar_input()).jaxpr


def operation_label(eqn: Any) -> str:
    """Format a JAX equation using its primitive name and relevant parameters."""
    primitive = eqn.primitive.name
    if primitive == "integer_pow":
        return f"{primitive}[y={eqn.params['y']}]"
    return primitive


def operation_display_label(eqn: Any) -> str:
    """Add JAX-recorded literal factors to a primitive label when present."""
    label = operation_label(eqn)
    literal_values = [
        _literal_label(variable.val)
        for variable in eqn.invars
        if hasattr(variable, "val")
    ]
    if literal_values and eqn.primitive.name != "integer_pow":
        return f"{label}[{', '.join(literal_values)}]"
    return label


def _literal_label(value: Any) -> str:
    """Format a scalar JAX literal without its implementation-specific wrapper."""
    try:
        return str(float(value))
    except (TypeError, ValueError):
        return str(value)


def operation_sequence(jaxpr: Any) -> list[str]:
    """Extract operation labels in recorded JAXPR order."""
    return [operation_label(eqn) for eqn in jaxpr.eqns]


def gradient_sequence_around_add_any(jaxpr: Any) -> list[str]:
    """Return the gradient operations surrounding the shared-node accumulation."""
    labels = operation_sequence(jaxpr)
    add_index = labels.index("add_any")
    start = max(0, add_index - 3)
    stop = min(len(labels), add_index + 2)
    return [f"{index:02d}: {labels[index]}" for index in range(start, stop)]


def _var_key(value: Any) -> int:
    return id(value)


def _variable_label(value: Any) -> str:
    return str(value)


def _graph_edges(jaxpr: Any) -> tuple[list[dict[str, str]], list[tuple[str, str]]]:
    """Convert JAXPR equations to operation nodes and data-flow edges."""
    nodes: list[dict[str, str]] = []
    edges: list[tuple[str, str]] = []
    producers: dict[int, str] = {}

    for input_index, variable in enumerate(jaxpr.invars):
        node_id = f"input_{input_index}"
        nodes.append({"id": node_id, "label": "Input\nr", "kind": "input"})
        producers[_var_key(variable)] = node_id

    for equation_index, equation in enumerate(jaxpr.eqns):
        node_id = f"op_{equation_index}"
        nodes.append(
            {
                "id": node_id,
                "label": f"{equation_index + 1:02d}  {operation_display_label(equation)}",
                "kind": "operation",
            }
        )
        for variable in equation.invars:
            source = producers.get(_var_key(variable))
            if source is not None:
                edges.append((source, node_id))
        for variable in equation.outvars:
            if not isinstance(variable, jax.core.DropVar):
                producers[_var_key(variable)] = node_id

    output_id = "output"
    nodes.append({"id": output_id, "label": "Output\nvalue", "kind": "output"})
    for variable in jaxpr.outvars:
        source = producers.get(_var_key(variable))
        if source is not None:
            edges.append((source, output_id))
    return nodes, edges


def _draw_graph(jaxpr: Any, title: str, output_path: Path) -> None:
    nodes, edges = _graph_edges(jaxpr)
    operation_nodes = [node for node in nodes if node["kind"] == "operation"]
    input_nodes = [node for node in nodes if node["kind"] == "input"]
    output_nodes = [node for node in nodes if node["kind"] == "output"]

    columns = 4 if len(operation_nodes) > 6 else len(operation_nodes)
    positions: dict[str, tuple[float, float]] = {}
    for index, node in enumerate(operation_nodes):
        row, column = divmod(index, columns)
        positions[node["id"]] = (column * 3.3 + 1.7, -row * 2.2)
    positions[input_nodes[0]["id"]] = (-1.0, 0.0)
    positions[output_nodes[0]["id"]] = (
        ((len(operation_nodes) - 1) % columns) * 3.3 + 4.1,
        -((len(operation_nodes) - 1) // columns) * 2.2,
    )

    figure_width = max(10.0, columns * 3.3 + 3.0)
    figure_height = max(4.0, ((len(operation_nodes) - 1) // columns + 1) * 2.6 + 1.5)
    figure, axis = plt.subplots(figsize=(figure_width, figure_height), constrained_layout=True)

    for source, target in edges:
        x_start, y_start = positions[source]
        x_end, y_end = positions[target]
        axis.annotate(
            "",
            xy=(x_end, y_end),
            xytext=(x_start, y_start),
            arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 1.1},
        )

    for node in nodes:
        x_position, y_position = positions[node["id"]]
        if node["kind"] == "operation":
            face_color = "#dceeff"
            width, height = 2.4, 0.9
        elif node["kind"] == "input":
            face_color = "#e2f0d9"
            width, height = 1.6, 0.9
        else:
            face_color = "#fce4d6"
            width, height = 1.6, 0.9
        patch = FancyBboxPatch(
            (x_position - width / 2, y_position - height / 2),
            width,
            height,
            boxstyle="round,pad=0.08",
            facecolor=face_color,
            edgecolor="#333333",
            linewidth=1.0,
        )
        axis.add_patch(patch)
        axis.text(x_position, y_position, node["label"], ha="center", va="center", fontsize=10)

    axis.set_title(title)
    axis.set_xlim(-2.0, max(position[0] for position in positions.values()) + 2.0)
    axis.set_ylim(min(position[1] for position in positions.values()) - 1.4, 1.4)
    axis.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, format="png")
    plt.close(figure)


def render_graphs(output_directory: Path) -> tuple[Path, Path]:
    """Render both graphs directly from their recorded JAXPRs."""
    graph_path = output_directory / "graph.png"
    gradient_path = output_directory / "grad-graph.png"
    _draw_graph(primal_jaxpr(), "JAX-recorded Lennard-Jones primal graph", graph_path)
    _draw_graph(
        gradient_jaxpr(),
        "JAX-recorded Lennard-Jones gradient graph",
        gradient_path,
    )
    return graph_path, gradient_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("artifacts/ad"),
        help="Directory for graph.png and grad-graph.png",
    )
    args = parser.parse_args()
    primal_recording = primal_jaxpr()
    gradient_recording = gradient_jaxpr()
    graph_path, gradient_path = render_graphs(args.output_directory)
    print("primal operations:", " -> ".join(operation_sequence(primal_recording)))
    print("add_any found:", "add_any" in operation_sequence(gradient_recording))
    print(
        "gradient operations around add_any:",
        " -> ".join(gradient_sequence_around_add_any(gradient_recording)),
    )
    print("generated:", graph_path)
    print("generated:", gradient_path)


if __name__ == "__main__":
    main()
