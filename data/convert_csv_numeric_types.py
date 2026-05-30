#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
把 CSV 中“看起来像数字的字符串”自动转换成数值。
- 整数 -> int
- 小数/科学计数法 -> Decimal（默认，避免精度丢失）
- 非数字字符串 -> 保持原样

为什么默认不用 float：
Python float 是二进制浮点，某些小数会有精度误差。
如果你要求“不能丢失任何精度”，小数最稳妥的做法是 Decimal。

注意：
CSV 格式本身不保存数据类型，它只保存文本。
所以：
1) 脚本在 Python 内存里会把值转成 int / Decimal
2) 再写回 CSV 时，文件里仍然是文本形式
3) 但这些文本会被规范化，后续 pandas / QGIS / Excel 更容易识别为数值
4) 如果你想真正保存“类型”，建议额外导出为 xlsx / parquet

用法示例：
python convert_csv_numeric_types.py input.csv
python convert_csv_numeric_types.py input.csv -o output.csv
python convert_csv_numeric_types.py input.csv --float-mode
"""

import csv
import re
import argparse
from decimal import Decimal, InvalidOperation
from pathlib import Path


# 支持：
# 123
# -123
# +123
# 123.45
# .45
# 123.
# 1e3
# -1.23E-4
NUMERIC_PATTERN = re.compile(
    r'^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$'
)

INT_PATTERN = re.compile(r'^[+-]?\d+$')


def convert_cell(value: str, float_mode: bool = False):
    """
    把单元格内容从字符串转换为数值：
    - 整数 -> int
    - 小数/科学计数法 -> Decimal（默认）或 float（float_mode=True）
    - 非数字 -> 原字符串
    """
    if value is None:
        return value

    s = value.strip()

    # 空字符串不处理
    if s == "":
        return value

    # 不是数字格式，原样返回
    if not NUMERIC_PATTERN.fullmatch(s):
        return value

    # 整数
    if INT_PATTERN.fullmatch(s):
        try:
            return int(s)
        except ValueError:
            return value

    # 小数 / 科学计数法
    if float_mode:
        # 注意：这里可能出现浮点精度误差
        try:
            return float(s)
        except ValueError:
            return value
    else:
        # Decimal 保证十进制精度
        try:
            return Decimal(s)
        except InvalidOperation:
            return value


def convert_csv(input_path: str, output_path: str = None, float_mode: bool = False):
    input_file = Path(input_path)
    if output_path is None:
        output_file = input_file.with_name(f"{input_file.stem}_converted{input_file.suffix}")
    else:
        output_file = Path(output_path)

    with input_file.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        raise ValueError("CSV 是空文件。")

    header = rows[0]
    data_rows = rows[1:]

    converted_rows = [header]

    for row in data_rows:
        converted_row = []
        for cell in row:
            converted_value = convert_cell(cell, float_mode=float_mode)

            # 写回 CSV 时：
            # int 直接写
            # Decimal 用普通十进制字符串写，避免科学计数法或精度变化
            if isinstance(converted_value, Decimal):
                converted_row.append(format(converted_value, 'f'))
            else:
                converted_row.append(converted_value)
        converted_rows.append(converted_row)

    with output_file.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(converted_rows)

    print(f"已完成：{output_file}")


def main():
    parser = argparse.ArgumentParser(description="自动转换 CSV 中的数字字符串")
    parser.add_argument("input_csv", help="输入 CSV 文件路径")
    parser.add_argument("-o", "--output", help="输出 CSV 文件路径", default=None)
    parser.add_argument(
        "--float-mode",
        action="store_true",
        help="把小数转成 Python float（可能有精度误差，不推荐）"
    )
    args = parser.parse_args()

    convert_csv(args.input_csv, args.output, args.float_mode)


if __name__ == "__main__":
    main()
