from __future__ import annotations
import json
import torch
from dataclasses import dataclass
from pathlib import Path
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from peft import get_peft_model
from transformers import AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer
from transformers import TrainingArguments







@dataclass
class TrainConfig:
    model_name: str = r"C:\\Users\\Dybala\\models\\Qwen2.5-0.5B-Instruct"

    train_file: str = "data/processed/train_sft_80.jsonl"
    output_dir: str = "outputs/qwen_financial_sentiment"
    num_train_epochs: int = 3 # 训练轮数
    learning_rate: float = 2e-4 # 学习率
    per_device_train_batch_size: int = 1 # 每个设备的训练批次大小
    max_length: int = 256

# 把 jsonl 训练文件读成 Python 里的样本列表
def load_samples(path: Path) -> list[dict]:
    samples = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
    return samples

# 把一条 SFT 样本拼成模型训练时实际看到的一段文本
def format_example(sample: dict) -> str:
    instruction = str(sample.get("instruction", "")).strip()
    input_text = str(sample.get("input", "")).strip()
    output_text = str(sample.get("output", "")).strip()

    return (
        f"{instruction}\n"
        f"新闻：{input_text}\n"
        f"答案：{output_text}"
    )

# 在正式训练前，先把前几条样本打印出来检查格式
def preview_samples(samples: list[dict], limit: int = 3) -> None:
    for index, sample in enumerate(samples[:limit], start=1):
        print(f"\n===== sample {index} =====")
        print(format_example(sample))

# 把原来的 instruction/input/output 样本，转换成只包含 text 字段的训练数据
def build_text_dataset(samples: list[dict]) -> list[dict]:
    dataset = []

    for sample in samples:
        dataset.append({"text": format_example(sample)})

    return dataset

# 把 Python 里的列表数据，转换成 HuggingFace 的 Dataset 对象
def to_hf_dataset(text_dataset: list[dict]) -> Dataset:
    return Dataset.from_list(text_dataset)

# 根据配置里的模型名，加载对应 tokenizer，并补上 pad_token（如果模型没有的话）
def load_tokenizer(config: TrainConfig):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

# 把一条 {"text": "..."} 样本，转换成 tokenizer 编码后的结果
def tokenize_function(example: dict, tokenizer, max_length: int) -> dict:
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )

# 根据配置加载基座大模型
def load_model(config: TrainConfig):
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        dtype=torch.float16,
        device_map="auto",
    )
    return model



# 定义 LoRA 微调参数
def build_lora_config() -> LoraConfig:
    return LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj", 
            "up_proj",
            "down_proj",
            "gate_proj",
        ],
    )


# 把 LoRA 配置真正挂到基座模型上
def apply_lora(model, lora_config: LoraConfig):
    return get_peft_model(model, lora_config)

# 创建训练参数对象，它会被 Trainer 用来控制训练过程
def build_training_arguments(config: TrainConfig, project_root: Path) -> TrainingArguments:
    output_dir = project_root / config.output_dir

    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        learning_rate=config.learning_rate,
        logging_steps=1,
        save_steps=10,
        save_total_limit=1,
        report_to="none",
        remove_unused_columns=False,
        gradient_accumulation_steps=4

    )

# 创建 batch 整理器，它会在训练时把多个样本整理成一个 batch，并自动处理 padding
def build_data_collator(tokenizer):
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

# 把模型、训练参数、训练集和 collator 组装成一个 Trainer
def build_trainer(model, training_args, train_dataset, data_collator):
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )



# 把前面写好的配置、数据读取、样本预览串起来，形成训练脚本的主流程骨架
def main() -> int:
    
    config = TrainConfig()
    project_root = Path(__file__).resolve().parent
    train_path = project_root / config.train_file
    samples = load_samples(train_path)
    text_dataset = build_text_dataset(samples)
    hf_dataset = to_hf_dataset(text_dataset)
    tokenizer = load_tokenizer(config)
    tokenized_dataset = hf_dataset.map(
        lambda example: tokenize_function(example, tokenizer, config.max_length),
         remove_columns=hf_dataset.column_names,
    )

    print(f"loaded {len(samples)} samples from {train_path}")
    preview_samples(samples)
    print(f"built text dataset with {len(text_dataset)} items")
    print(text_dataset[0]["text"])
    print(hf_dataset)
    print(tokenizer.__class__.__name__)
    print(tokenizer.pad_token)
    print(tokenized_dataset)
    print(tokenized_dataset[0]["input_ids"][:20])
    print(tokenized_dataset[0]["attention_mask"][:20])

    lora_config = build_lora_config()
    print(lora_config)

    model = load_model(config)
    model = apply_lora(model, lora_config)
    model.print_trainable_parameters()

    training_args = build_training_arguments(config, project_root)
    data_collator = build_data_collator(tokenizer)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    trainer = build_trainer(model, training_args, tokenized_dataset, data_collator)

    print(training_args)
    print(trainer)
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)

    """读取 tokenized dataset
构建 batch
前向传播
计算 loss
反向传播
更新 LoRA 参数"""


    return 0





"""读取训练配置
读取训练样本
打印样本数量
预览前几条样本"""

if __name__ == "__main__":
    raise SystemExit(main())
