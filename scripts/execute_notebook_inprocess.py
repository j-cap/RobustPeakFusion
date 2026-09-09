"""Execute a notebook without launching a socket-based Jupyter kernel.

This helper is useful in restricted CI/sandbox environments. Standard users can
run the notebook normally in JupyterLab. It supports the output types used by
this project's notebook: text, rich HTML representations, and Matplotlib PNGs.
"""

from __future__ import annotations

import argparse
import ast
import base64
import contextlib
import io
import os
from pathlib import Path
import traceback

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
import nbformat


def execute_cell(source: str, namespace: dict) -> tuple[object | None, str]:
    tree = ast.parse(source, mode="exec")
    final_expression = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        final_expression = ast.Expression(tree.body.pop().value)
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        if tree.body:
            exec(compile(tree, "<notebook-cell>", "exec"), namespace)
        value = eval(compile(final_expression, "<notebook-cell>", "eval"), namespace) if final_expression else None
    return value, stream.getvalue()


def rich_output(value: object) -> dict | None:
    if value is None:
        return None
    data = {"text/plain": repr(value)}
    html = getattr(value, "_repr_html_", None)
    if callable(html):
        rendered = html()
        if rendered is not None:
            data["text/html"] = rendered
    return nbformat.v4.new_output("execute_result", data=data)


def figure_outputs() -> list[dict]:
    outputs = []
    for number in plt.get_fignums():
        figure = plt.figure(number)
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        outputs.append(nbformat.v4.new_output("display_data", data={"image/png": encoded}))
    plt.close("all")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path)
    args = parser.parse_args()
    notebook = nbformat.read(args.notebook, as_version=4)
    namespace = {"__name__": "__main__"}
    execution_count = 0

    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        execution_count += 1
        cell.execution_count = execution_count
        cell.outputs = []
        try:
            value, stream = execute_cell(cell.source, namespace)
            if stream:
                cell.outputs.append(nbformat.v4.new_output("stream", name="stdout", text=stream))
            result = rich_output(value)
            if result:
                result.execution_count = execution_count
                cell.outputs.append(result)
            cell.outputs.extend(figure_outputs())
        except Exception as exc:
            trace = traceback.format_exc().splitlines()
            cell.outputs.append(
                nbformat.v4.new_output(
                    "error",
                    ename=type(exc).__name__,
                    evalue=str(exc),
                    traceback=trace,
                )
            )
            nbformat.write(notebook, args.notebook)
            raise RuntimeError(f"notebook cell {index + 1} failed") from exc

    nbformat.write(notebook, args.notebook)
    print(f"Executed {execution_count} code cells in {args.notebook}")


if __name__ == "__main__":
    main()

