#!/usr/bin/env python3

from pathlib import Path
import re

template = Path(
    "run_adaptive_c1p71875_pH8p25_nacl100_seed20260844.sh"
)

if not template.exists():
    raise SystemExit(f"ERROR：找不到模板脚本：{template}")

template_text = template.read_text(encoding="utf-8")

required = [
    "c1p71875",
    "1.71875",
    "20260844",
]

for token in required:
    if token not in template_text:
        raise SystemExit(f"ERROR：模板中缺少参数：{token}")

tasks = [
    {
        "concentration": "0.3125",
        "code": "0p3125",
        "seed": "20260845",
        "banner": "C0P3125",
    },
    {
        "concentration": "0.625",
        "code": "0p625",
        "seed": "20260846",
        "banner": "C0P625",
    },
    {
        "concentration": "0.9375",
        "code": "0p9375",
        "seed": "20260847",
        "banner": "C0P9375",
    },
    {
        "concentration": "1.640625",
        "code": "1p640625",
        "seed": "20260848",
        "banner": "C1P640625",
    },
]

created = []

for task in tasks:
    dst = Path(
        f'run_adaptive_c{task["code"]}_'
        f'pH8p25_nacl100_seed{task["seed"]}.sh'
    )

    text = template_text

    # 修改任务名、输出目录及状态名中的浓度编码
    text = text.replace(
        "c1p71875",
        f'c{task["code"]}',
    )

    # 修改独立的浓度数值，不会误改 pH=8.25
    text, count = re.subn(
        r"(?<![\d.])1\.71875(?![\d.])",
        task["concentration"],
        text,
    )

    if count == 0:
        raise SystemExit(
            f'ERROR：没有替换到浓度值：{task["concentration"]}'
        )

    # 修改随机种子
    text = text.replace(
        "20260844",
        task["seed"],
    )

    # 修改完成横幅
    text = text.replace(
        "ADAPTIVE_C1P71875_PH8P25_NACL100",
        f'ADAPTIVE_{task["banner"]}_PH8P25_NACL100',
    )

    # 检查旧参数是否残留
    forbidden = [
        "c1p71875",
        "20260844",
        "ADAPTIVE_C1P71875_PH8P25_NACL100",
    ]

    leftovers = [x for x in forbidden if x in text]

    if leftovers:
        raise SystemExit(
            f"ERROR：{dst} 中仍残留旧参数：{leftovers}"
        )

    dst.write_text(text, encoding="utf-8")
    dst.chmod(0o755)
    created.append(dst)

    print(
        f'生成：{dst}  '
        f'c={task["concentration"]}  '
        f'seed={task["seed"]}'
    )

print(f"\n共生成 {len(created)} 个脚本")
