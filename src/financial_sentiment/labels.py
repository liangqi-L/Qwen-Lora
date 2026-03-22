LABELS = ("利好", "中性", "利空")


def validate_label(label: str) -> str:
    if label not in LABELS:
        raise ValueError(f"invalid label: {label}")
    return label
