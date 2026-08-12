#!/usr/bin/env python3
"""Build an orthorhombic slab runner from the validated cubic HOOMD runner."""

from __future__ import annotations

import ast
import re
from pathlib import Path


SOURCE = Path("scripts/run_single_state_hoomd.py")
OUTPUT = Path("scripts/run_direct_coexistence_hoomd.py")

if not SOURCE.exists():
    raise FileNotFoundError(SOURCE)

text = SOURCE.read_text(encoding="utf-8")


def regex_patch(
    pattern: str,
    replacement: str,
    label: str,
) -> None:
    global text

    updated, count = re.subn(
        pattern,
        replacement,
        text,
        count=1,
        flags=(
            re.DOTALL
            | re.MULTILINE
            | re.VERBOSE
        ),
    )

    if count != 1:
        raise RuntimeError(
            f"Patch failed for {label}: matched {count} times."
        )

    text = updated
    print("PATCH PASS:", label)


def patch_wrapped_reduced_with_ast() -> None:
    """Replace the wrapped_reduced assignment by AST line location."""

    global text

    tree = ast.parse(text)

    target_node: ast.Assign | ast.AnnAssign | None = None
    expression_node: ast.AST | None = None

    for node in ast.walk(tree):
        target_name = None
        value = None

        if isinstance(node, ast.Assign):
            if len(node.targets) != 1:
                continue

            target = node.targets[0]

            if isinstance(target, ast.Name):
                target_name = target.id

            value = node.value

        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                target_name = node.target.id

            value = node.value

        if target_name != "wrapped_reduced":
            continue

        if value is None:
            continue

        # Expected form: np.mod(expression, box_length_reduced)
        if isinstance(value, ast.Call):
            function = value.func

            if (
                isinstance(function, ast.Attribute)
                and function.attr in {"mod", "remainder"}
                and len(value.args) >= 1
            ):
                target_node = node
                expression_node = value.args[0]
                break

        # Also support: expression % box_length_reduced
        if (
            isinstance(value, ast.BinOp)
            and isinstance(value.op, ast.Mod)
        ):
            target_node = node
            expression_node = value.left
            break

    if target_node is None or expression_node is None:
        raise RuntimeError(
            "AST could not locate a supported "
            "wrapped_reduced assignment."
        )

    if (
        target_node.end_lineno is None
        or expression_node.end_lineno is None
    ):
        raise RuntimeError(
            "Python AST does not provide end line numbers."
        )

    expression = ast.get_source_segment(
        text,
        expression_node,
    )

    if expression is None:
        raise RuntimeError(
            "Could not recover the wrapped coordinate expression."
        )

    lines = text.splitlines(keepends=True)

    start = target_node.lineno - 1
    end = target_node.end_lineno

    original_first_line = lines[start]
    indent = original_first_line[
        : len(original_first_line)
        - len(original_first_line.lstrip())
    ]

    expression_lines = expression.splitlines()

    if len(expression_lines) == 1:
        expression_block = (
            f"{indent}        {expression_lines[0].strip()}\n"
        )
    else:
        expression_block = "".join(
            f"{indent}        {line.rstrip()}\n"
            for line in expression_lines
        )

    replacement = (
        f"{indent}wrapped_reduced = np.mod(\n"
        f"{indent}    (\n"
        f"{expression_block}"
        f"{indent}    )\n"
        f"{indent}    + 0.5 * box_lengths_reduced,\n"
        f"{indent}    box_lengths_reduced,\n"
        f"{indent})\n"
    )

    lines[start:end] = [replacement]
    text = "".join(lines)

    print("PATCH PASS: wrap orthorhombic trajectory coordinates")
    print("  Original expression:", expression.replace("\n", " "))


regex_patch(
    r'''
    ^(?P<indent>[ \t]*)
    box_length_reduced
    \s*=\s*
    float\(
        \s*data\["box_length_reduced"\]\s*
    \)
    ''',
    '''\\g<indent>box_lengths_reduced = np.asarray(
\\g<indent>    data["box_lengths_reduced"],
\\g<indent>    dtype=np.float64,
\\g<indent>)
\\g<indent>
\\g<indent>if box_lengths_reduced.shape != (3,):
\\g<indent>    raise ValueError(
\\g<indent>        "box_lengths_reduced must have shape (3,)."
\\g<indent>    )
\\g<indent>
\\g<indent># Compatibility alias for inherited diagnostics.
\\g<indent>box_length_reduced = float(
\\g<indent>    box_lengths_reduced[0]
\\g<indent>)''',
    "load reduced orthorhombic box",
)

regex_patch(
    r'''
    ^(?P<indent>[ \t]*)
    box_length_nm
    \s*=\s*
    float\(
        \s*metadata\["box_length_nm"\]\s*
    \)
    ''',
    '''\\g<indent>box_lengths_nm = np.asarray(
\\g<indent>    metadata["box_lengths_nm"],
\\g<indent>    dtype=np.float64,
\\g<indent>)
\\g<indent>
\\g<indent>if box_lengths_nm.shape != (3,):
\\g<indent>    raise ValueError(
\\g<indent>        "box_lengths_nm must have shape (3,)."
\\g<indent>    )
\\g<indent>
\\g<indent># Compatibility alias for inherited diagnostics.
\\g<indent>box_length_nm = float(
\\g<indent>    box_lengths_nm[0]
\\g<indent>)''',
    "load physical orthorhombic box",
)

regex_patch(
    r'''
    ^(?P<indent>[ \t]*)
    snapshot\.configuration\.box
    \s*=\s*
    \[
        \s*box_length_reduced\s*,
        \s*box_length_reduced\s*,
        \s*box_length_reduced\s*,
        \s*0\.0\s*,
        \s*0\.0\s*,
        \s*0\.0\s*,?
    \s*\]
    ''',
    '''\\g<indent>snapshot.configuration.box = [
\\g<indent>    float(box_lengths_reduced[0]),
\\g<indent>    float(box_lengths_reduced[1]),
\\g<indent>    float(box_lengths_reduced[2]),
\\g<indent>    0.0,
\\g<indent>    0.0,
\\g<indent>    0.0,
\\g<indent>]''',
    "set orthorhombic HOOMD snapshot box",
)

patch_wrapped_reduced_with_ast()

regex_patch(
    r'''
    ^(?P<indent>[ \t]*)
    box_length_nm
    \s*=\s*
    np\.float64\(
        \s*box_length_nm\s*
    \)
    ''',
    '''\\g<indent>box_lengths_nm=np.asarray(
\\g<indent>    box_lengths_nm,
\\g<indent>    dtype=np.float64,
\\g<indent>)''',
    "write trajectory orthorhombic box",
)

regex_patch(
    r'''
    ^(?P<indent>[ \t]*)
    box_length_reduced
    \s*=\s*
    np\.float64\(
        \s*box_length_reduced\s*
    \)
    ''',
    '''\\g<indent>box_lengths_reduced=np.asarray(
\\g<indent>    box_lengths_reduced,
\\g<indent>    dtype=np.float64,
\\g<indent>)''',
    "write final-state orthorhombic box",
)

if "RUN_SINGLE_STATE_HOOMD: PASS" not in text:
    raise RuntimeError(
        "Could not find the ordinary-runner PASS marker."
    )

text = text.replace(
    "RUN_SINGLE_STATE_HOOMD: PASS",
    "RUN_DIRECT_COEXISTENCE_HOOMD: PASS",
    1,
)

# Validate the generated Python source before saving.
ast.parse(text)

OUTPUT.write_text(
    text,
    encoding="utf-8",
)

print()
print("Source :", SOURCE)
print("Output :", OUTPUT)
print("Generated source parsed successfully.")
print("BUILD_ORTHORHOMBIC_HOOMD_RUNNER: PASS")
