PUNCTUATIONS = set("，。！？；：、,.!?;:（）()“”\"'《》【】[]{} \t\r\n")


def char_tokenize(text):
    return [ch for ch in str(text) if ch not in PUNCTUATIONS]


def print_tokenization_examples():
    examples = [
        "今天真的太开心了，感觉一切都很顺利",
        "这件事太过分了，我真的很生气",
        "这个东西真的很恶心，看着就反感",
        "今天心情很低落，感觉有点难过",
    ]

    print("\n====== 字符级切分示例 ======")
    for text in examples:
        print(f"\n文本: {text}")
        print(f"char_tokenize: {char_tokenize(text)}")
