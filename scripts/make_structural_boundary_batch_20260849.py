#!/usr/bin/env python3

from pathlib import Path
import re

template = Path(
    "run_adaptive_c1p71875_pH8p25_nacl100_seed20260844.sh"
)

if not template.exists():
    raise SystemExit(
        f"ERROR：找不到模板脚本：{template}\n"
        "请先执行：ls -lh run_adaptive_c1p71875*"
    )

template_text = template.read_text(encoding="utf-8")

required_tokens = [
    "c1p71875",
    "1.71875",
    "20260844",
]

for token in required_tokens:
    if token not in template_text:
        raise SystemExit(
            f"ERROR：模板脚本中找不到必要参数：{token}"
        )

tasks = [
    {
        "concentration": "6.25",
        "code": "6p25",
        "seed": "20260849",
        "banner": "C6P25",
    },
    {
        "concentration": "7.5",
        "code": "7p5",
        "seed": "20260850",
        "banner": "C7P5",
    },
    {
        "concentration": "8.75",
        "code": "8p75",
        "seed": "20260851",
        "banner": "C8P75",
    },
]

created = []

for task in tasks:
    dst = Path(
        f'run_adaptive_c{task["code"]}_'
        f'pH8p25_nacl100_seed{task["seed"]}.sh'
    )

    text = template_text

    # 修改运行ID、结果目录和文件名中的浓度编码
    text = text.replace(
        "c1p71875",
        f'c{task["code"]}',
    )

    # 修改独立出现的浓度值，不影响 pH=8.25
    text, replacements = re.subn(
        r"(?<![\d.])1\.71875(?![\d.])",
        task["concentration"],
        text,
    )

    if replacements == 0:
        raise SystemExit(
            f'ERROR：没有替换到浓度 {task["concentration"]}'
        )

    # 修改随机种子
    text = text.replace(
        "20260844",
        task["seed"],
    )

    # 修改终端完成横幅
    text = text.replace(
        "ADAPTIVE_C1P71875_PH8P25_NACL100",
        f'ADAPTIVE_{task["banner"]}_PH8P25_NACL100',
    )

    forbidden = [
        "c1p71875",
        "20260844",
        "ADAPTIVE_C1P71875_PH8P25_NACL100",
    ]

    leftovers = [
        token for token in forbidden
        if token in text
    ]

    if leftovers:
        raise SystemExit(
            f"ERROR：{dst} 中仍有旧参数：{leftovers}"
        )

    dst.write_text(text, encoding="utf-8")
    dst.chmod(0o755)

    created.append(dst)

    print(
        f'生成：{dst}\n'
        f'  concentration = {task["concentration"]}\n'
        f'  seed          = {task["seed"]}'
    )

print(f"\n成功生成 {len(created)} 个脚本。")
