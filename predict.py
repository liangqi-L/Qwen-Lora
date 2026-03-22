from __future__ import annotations

import torch
from peft import PeftModel
"""这个是关键
因为你现在推理时不是直接加载一个完整新模型，而是：

先加载原始基座模型
再把 LoRA adapter 挂上去"""
from transformers import AutoModelForCausalLM
# 加载原始基座模型
from transformers import AutoTokenizer
# 加载 tokenizer

# 把推理需要的两个核心对象一起加载出来：模型 和 tokenizer
def load_model_and_tokenizer():
    base_model_path = r"C:\Users\Dybala\models\Qwen2.5-0.5B-Instruct"
    adapter_path = r"C:\Users\Dybala\Desktop\quant\llm_financial_sentiment_project\outputs\qwen_financial_sentiment"

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        dtype=torch.float16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, adapter_path)

    return model, tokenizer

# 把你输入的一条新闻，拼成模型推理时要看的提示词
def build_prompt(news_text: str) -> str:
    return (
        "请判断下面这条金融新闻对相关股票或市场情绪的影响，"
        "只输出一个标签：利好、中性、利空。\n"
        f"新闻：{news_text}\n"
        "答案："
    )

# 输入一条新闻，调用模型生成预测结果，并把结果字符串提取出来
def predict_sentiment(model, tokenizer, news_text: str) -> str:
    prompt = build_prompt(news_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = generated_text[len(prompt):].strip()

    for label in ("利好", "中性", "利空"):
        if label in answer:
            return label

    return answer


# 把推理流程真正串起来，形成一个可以直接运行的脚本入口
def main() -> int:
    model, tokenizer = load_model_and_tokenizer()

    news_text = input("请输入一条金融新闻：").strip()
    prediction = predict_sentiment(model, tokenizer, news_text)

    print("预测结果：", prediction)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
