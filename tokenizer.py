import random
import time


PUNCTUATIONS = set("，。！？；：、,.!?;:（）()“”\"'《》【】[]{} \t\r\n")

NEGATION_WORDS = [
    "不是", "没有", "没", "不", "别", "无", "难以", "无法", "没法", "不太",
]

DEGREE_WORDS = [
    "非常", "特别", "真的", "很", "太", "有点", "一点", "比较", "格外",
    "十分", "超级", "明显", "一下子", "直接", "慢慢", "越来越",
]

COMMON_WORDS = [
    "微博", "今天", "朋友", "同学", "老师", "家人", "工作", "学习",
    "考试", "成绩", "电影", "电视剧", "音乐", "游戏", "评论",
    "消息", "事情", "生活", "心情", "感觉", "真的", "非常",
    "特别", "有点", "不是", "没有", "不太", "为什么", "回复",
    "转发", "支持", "希望", "喜欢", "讨厌", "觉得", "看到",
    "知道", "突然", "现在", "以后", "大家", "现场", "气氛",
]

JOY_WORDS = [
    "开心", "高兴", "快乐", "喜欢", "满意", "幸福", "惊喜", "顺利",
    "开心死了", "太开心", "好喜欢", "棒", "不错", "舒服",
    "开心坏了", "笑死", "哈哈", "哈哈哈", "激动", "感动",
    "甜", "温暖", "美好", "可爱", "赞", "恭喜", "快乐",
]

ANGER_WORDS = [
    "生气", "愤怒", "恼火", "烦躁", "窝火", "火气", "火大",
    "气死", "气炸", "气得不行", "太过分", "离谱", "受不了",
    "忍不了", "可恶", "烦死", "怒", "骂", "投诉", "恶劣",
    "动怒", "失控", "不爽",
]

DISGUST_WORDS = [
    "恶心", "讨厌", "厌恶", "反感", "嫌弃", "无语", "下头",
    "垃圾", "恶臭", "恶俗", "难闻", "脏", "丑", "烂", "油腻",
    "膈应", "受够了", "烦人", "不想看", "不想理", "龌龊",
    "变态", "恶搞",
]

LOW_WORDS = [
    "难过", "伤心", "低落", "失望", "失落", "委屈", "沮丧",
    "悲伤", "难受", "心酸", "心累", "累了", "崩溃", "哭了",
    "想哭", "遗憾", "孤独", "无助", "压抑", "不开心",
    "心情不好", "郁闷", "悲剧", "可怜", "舍不得", "怕",
]

JOY_PHRASES = [
    "真的很开心", "太开心了", "非常开心", "特别开心", "好喜欢",
    "开心死了", "幸福感爆棚", "笑死我了", "真的很好", "太棒了",
    "中秋节快乐", "万圣节快乐", "好开心哦",
]

ANGER_PHRASES = [
    "真的很生气", "气死我了", "气得不行", "太过分了", "太离谱了",
    "真的受不了", "烦死了", "火气上来了", "忍不了了",
    "有点火大", "特别容易动怒",
]

DISGUST_PHRASES = [
    "真的很恶心", "太恶心了", "看着反胃", "真的下头",
    "太下头了", "非常讨厌", "特别反感", "恶心死了",
    "不想再看", "不要脸", "真恶心",
]

LOW_PHRASES = [
    "真的很难过", "有点低落", "心情不好", "真的很失望",
    "有点失望", "挺失落的", "心里难受", "想哭了",
    "真的很委屈", "感觉很累", "真遗憾", "好郁闷",
]


def make_phrase_terms():
    phrase_terms = set(JOY_PHRASES + ANGER_PHRASES + DISGUST_PHRASES + LOW_PHRASES)
    emotion_word_groups = [JOY_WORDS, ANGER_WORDS, DISGUST_WORDS, LOW_WORDS]

    for degree_word in DEGREE_WORDS:
        for word_group in emotion_word_groups:
            for emotion_word in word_group:
                phrase_terms.add(degree_word + emotion_word)

    for negation_word in NEGATION_WORDS:
        for word_group in emotion_word_groups:
            for emotion_word in word_group:
                phrase_terms.add(negation_word + emotion_word)

        for degree_word in DEGREE_WORDS:
            for word_group in emotion_word_groups:
                for emotion_word in word_group:
                    phrase_terms.add(negation_word + degree_word + emotion_word)

    return sorted(phrase_terms, key=len, reverse=True)


PHRASE_TERMS = make_phrase_terms()
WORD_TERMS = sorted(
    set(
        COMMON_WORDS
        + JOY_WORDS
        + ANGER_WORDS
        + DISGUST_WORDS
        + LOW_WORDS
        + DEGREE_WORDS
        + NEGATION_WORDS
    ),
    key=len,
    reverse=True,
)


def longest_match_tokenize(text, terms):
    text = str(text)
    tokens = []
    i = 0

    while i < len(text):
        if text[i] in PUNCTUATIONS:
            i += 1
            continue

        matched = None
        for term in terms:
            if text.startswith(term, i):
                matched = term
                break

        if matched is not None:
            tokens.append(matched)
            i += len(matched)
        else:
            tokens.append(text[i])
            i += 1

    return tokens


class TrieNode:
    def __init__(self):
        self.children = {}
        self.term = None


class TrieTokenizer:
    def __init__(self, terms, punctuations):
        self.root = TrieNode()
        self.punctuations = punctuations
        for term in terms:
            self.insert(term)

    def insert(self, term):
        if not term:
            return
        node = self.root
        for ch in str(term):
            node = node.children.setdefault(ch, TrieNode())
        node.term = str(term)

    def tokenize(self, text):
        text = str(text)
        tokens = []
        i = 0

        while i < len(text):
            if text[i] in self.punctuations:
                i += 1
                continue

            node = self.root
            j = i
            matched = None
            matched_end = i

            while j < len(text):
                ch = text[j]
                if ch not in node.children:
                    break
                node = node.children[ch]
                j += 1
                if node.term is not None:
                    matched = node.term
                    matched_end = j

            if matched is not None:
                tokens.append(matched)
                i = matched_end
            else:
                tokens.append(text[i])
                i += 1

        return tokens


_TRIE_BUILD_START = time.perf_counter()
WORD_TRIE_TOKENIZER = TrieTokenizer(WORD_TERMS, PUNCTUATIONS)
PHRASE_TRIE_TOKENIZER = TrieTokenizer(PHRASE_TERMS, PUNCTUATIONS)
print(f"[TIME] build Trie cost: {time.perf_counter() - _TRIE_BUILD_START:.4f}s", flush=True)


def char_tokenize(text):
    return [ch for ch in str(text) if ch not in PUNCTUATIONS]


def word_tokenize(text):
    return WORD_TRIE_TOKENIZER.tokenize(text)


def phrase_tokenize(text):
    return PHRASE_TRIE_TOKENIZER.tokenize(text)


def test_trie_tokenizer_consistency(texts=None, sample_size=100, seed=42):
    if texts is None:
        texts = [
            "今天真的太开心了，感觉一切都很顺利",
            "这件事太过分了，我真的很生气",
            "这个东西真的很恶心，看着就反感",
            "今天心情很低落，感觉有点难过",
        ]

    texts = [str(text) for text in texts]
    if len(texts) > sample_size:
        rng = random.Random(seed)
        texts = rng.sample(texts, sample_size)

    mismatches = []
    for text in texts:
        old_word = longest_match_tokenize(text, WORD_TERMS)
        new_word = WORD_TRIE_TOKENIZER.tokenize(text)
        old_phrase = longest_match_tokenize(text, PHRASE_TERMS)
        new_phrase = PHRASE_TRIE_TOKENIZER.tokenize(text)

        if old_word != new_word:
            mismatches.append({
                "text": text,
                "tokenizer": "word",
                "old": old_word,
                "trie": new_word,
            })
        if old_phrase != new_phrase:
            mismatches.append({
                "text": text,
                "tokenizer": "phrase",
                "old": old_phrase,
                "trie": new_phrase,
            })

    if mismatches:
        print("Trie tokenizer consistency failed. First 5 mismatches:")
        for item in mismatches[:5]:
            print(f"text: {item['text']}")
            print(f"tokenizer: {item['tokenizer']}")
            print(f"old:  {item['old']}")
            print(f"trie: {item['trie']}")
        raise AssertionError(f"Trie tokenizer mismatches: {len(mismatches)}")

    print(f"Trie tokenizer consistency passed on {len(texts)} texts.", flush=True)


def print_tokenization_examples():
    examples = [
        "今天真的太开心了，感觉一切都很顺利",
        "这件事太过分了，我真的很生气",
        "这个东西真的很恶心，看着就反感",
        "今天心情很低落，感觉有点难过",
    ]

    print("\n====== 三路切分示例 ======")
    for text in examples:
        print(f"\n文本: {text}")
        print(f"char_tokenize: {char_tokenize(text)}")
        print(f"word_tokenize: {word_tokenize(text)}")
        print(f"phrase_tokenize: {phrase_tokenize(text)}")
