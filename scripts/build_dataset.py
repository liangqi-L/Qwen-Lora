from __future__ import annotations
"""这一行的作用：

让类型注解的行为更稳定
是现代 Python 项目里很常见的一行
现在你不用深究原理，先把它当成“类型标注兼容增强”"""

import json
import sys
"""为什么需要这个？
因为我们这个项目里自己的代码放在 src/financial_sentiment/ 下面。
直接运行 scripts/build_dataset.py 时，
Python 默认不一定能找到这个包，所以要手动把 src 目录加进去。"""
from pathlib import Path
"""后面我们会用它来表示：
项目根目录
src 目录
原始数据文件路径
输出文件路径"""


PROJECT_ROOT = Path(__file__).resolve().parents[1]#找到“项目根目录”在哪里
"""__file__：当前这个脚本自己的路径
Path(__file__)：把它变成 Path 对象
.resolve()：得到绝对路径
.parents[1]：向上回退两层目录"""

SRC_DIR = PROJECT_ROOT / "src" # 基于刚才的项目根目录，继续定位到 src 文件夹

if str(SRC_DIR) not in sys.path:# 检查 SRC_DIR 这个路径，当前是不是已经在 Python 的模块搜索路径里
    sys.path.insert(0, str(SRC_DIR))


from financial_sentiment.labels import validate_label


INSTRUCTION = "请判断下面这条金融新闻对相关股票或市场情绪的影响，只输出一个标签：利好、中性、利空。"


def load_raw_data(path: Path) -> list[dict]:# > list[dict] 表示函数最后会返回“字典组成的列表”
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("raw data must be a JSON list")

    return data

# 提取标题和正文，拼成输入文本
def build_input_text(sample: dict) -> str:
    title = str(sample.get("title", "")).strip()
    content = str(sample.get("content", "")).strip()
    return f"标题：{title}\n正文：{content}"

# 定义一个函数，专门把“一条原始新闻样本”转换成“一条微调样本”
def convert_one_sample(sample: dict) -> dict:
    label = validate_label(str(sample["label"]).strip())
    """这一行很关键，作用是：

从原始样本里取出 label
转成字符串
去掉空格
调用 validate_label(...) 检查是否合法
最后把结果存到变量 label"""

    input_text = build_input_text(sample)

    return {
        "instruction": INSTRUCTION,
        "input": input_text,
        "output": label,
    }

"""这一行的作用：

定义一个函数，负责把很多条样本保存成 jsonl 文件
samples: list[dict] 表示输入是“字典列表”
path: Path 表示输出位置是一个路径
-> None 表示这个函数不返回结果，它只是执行保存动作"""
def save_jsonl(samples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for sample in samples:
            line = json.dumps(sample, ensure_ascii=False)
            file.write(line + "\n")


def main() -> int:
    input_path = PROJECT_ROOT / "data" / "raw" / "news_samples.json"
    # 最终它指向的是这个文件：news_samples.json  raw=原始数据 processed=处理后数据  train_sft=训练用的微调数据  jsonl=每行一个 JSON 对象的文本文件
    output_path = PROJECT_ROOT / "data" / "processed" / "train_sft.jsonl"

    raw_data = load_raw_data(input_path)
    converted_data = [convert_one_sample(sample) for sample in raw_data]
    """把 raw_data 里的每一条原始样本
都送进 convert_one_sample(sample)
转成标准微调样本
最后收集成一个新的列表 converted_data"""
    save_jsonl(converted_data, output_path)

    print(f"converted {len(converted_data)} samples")
    print(f"saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
