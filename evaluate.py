from __future__ import annotations

import json
from pathlib import Path

from predict import load_model_and_tokenizer
from predict import predict_sentiment


LABELS = ("利好", "中性", "利空")


def load_test_samples(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("test data must be a JSON list")

    return data


def evaluate(samples: list[dict]) -> dict:
    model, tokenizer = load_model_and_tokenizer()

    total = len(samples)
    correct = 0
    valid = 0
    results = []

    for sample in samples:
        text = str(sample["text"]).strip()
        gold = str(sample["label"]).strip()
        pred = predict_sentiment(model, tokenizer, text).strip()

        if pred in LABELS:
            valid += 1

        if pred == gold:
            correct += 1

        results.append(
            {
                "text": text,
                "gold": gold,
                "pred": pred,
                "correct": pred == gold,
            }
        )

    accuracy = correct / total if total else 0.0
    valid_rate = valid / total if total else 0.0

    return {
        "total": total,
        "correct": correct,
        "valid": valid,
        "accuracy": accuracy,
        "valid_rate": valid_rate,
        "results": results,
    }


def main() -> int:
    project_root = Path(__file__).resolve().parent
    test_path = project_root / "data" / "raw" / "test_samples_20.json"

    samples = load_test_samples(test_path)
    report = evaluate(samples)

    print(f"total: {report['total']}")
    print(f"correct: {report['correct']}")
    print(f"valid: {report['valid']}")
    print(f"accuracy: {report['accuracy']:.4f}")
    print(f"valid_rate: {report['valid_rate']:.4f}")

    for item in report["results"]:
        print("-" * 50)
        print("text:", item["text"])
        print("gold:", item["gold"])
        print("pred:", item["pred"])
        print("correct:", item["correct"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
