PUNCTUATIONS = set("，。！？；：、,.!?;:（）()“”\"'《》【】[] \t\r\n")

NEGATION_WORDS = [
    "不是", "没有", "没", "不", "别", "无", "难以", "无法", "没法"
]

DEGREE_WORDS = [
    "非常", "特别", "真的", "很", "太", "有点", "一点", "比较", "格外",
    "十分", "超级", "明显", "一下子", "直接", "慢慢", "忍不住"
]

GENERAL_WORDS = [
    "这个", "电影", "快递", "状态", "正在", "派送", "会议", "表格",
    "系统", "提示", "重新登录", "课程", "资料", "上传", "文件", "桌面", "公交",
    "成绩", "考试", "面试", "比赛", "作业", "朋友", "同学", "老师", "同事",
    "队友", "妈妈", "大家", "项目", "计划", "旅行", "生日", "礼物", "消息",
    "表扬", "感谢", "顺利", "完成", "通过", "取消", "拒绝", "误会", "离别",
    "照片", "电话", "客服", "外卖", "责任", "退款", "插队", "打断", "限制",
    "通知", "安排", "记录", "确认", "更新", "处理", "流程", "情绪", "心情",
    "态度", "结果", "预期", "想象", "期待", "感觉", "今天", "收到"
]

HAPPY_WORDS = [
    "开心", "高兴", "快乐", "喜欢", "满意", "幸福", "轻松", "惊喜", "顺利",
    "愉快", "开心得不行", "幸福感", "有动力", "心情好", "非常开心"
]

ANGER_WORDS = [
    "生气", "愤怒", "恼火", "烦躁", "窝火", "火气", "火大", "气死",
    "气得不行", "离谱", "太过分", "受不了", "忍不了", "恶心",
    "投诉", "骂", "不满", "满肚子火"
]

DISAPPOINTMENT_WORDS = [
    "失望", "失落", "难过", "低落", "委屈", "难受", "悲伤", "心凉",
    "遗憾", "不是很好", "不太好", "没有想象中好", "不如预期",
    "期待落空", "有点失望", "有点难过", "有点委屈"
]

NEUTRAL_WORDS = [
    "平静", "普通", "正常", "稳定", "记录", "确认", "更新", "派送",
    "安排", "流程", "通知", "同步", "处理", "显示", "状态"
]

GENERAL_NEGATIVE_WORDS = [
    "不好", "不好看", "差", "一般", "不满意", "不舒服", "不太行"
]

ANGER_PHRASES = [
    "真的很生气", "非常生气", "气得不行", "气死我了", "太离谱了",
    "太过分了", "真的很恼火", "心里特别窝火", "受不了了", "忍不了了",
    "满肚子火", "火气一下子上来了", "客服态度太离谱",
    "一肚子火", "越想越生气"
]

DISAPPOINTMENT_PHRASES = [
    "有点失望", "真的有点失望", "挺失望的", "不是很好", "不太好",
    "没有想象中好", "没有预期好", "不如预期", "本来很期待",
    "结果有点失望", "心里有点失落", "感觉挺遗憾", "有点难过",
    "有点委屈", "心情低落", "没有那么好", "期待落空",
    "心里空落落", "情绪低了"
]

HAPPY_PHRASES = [
    "真的非常开心", "非常开心", "特别开心", "开心得不行", "心情特别好",
    "幸福感爆棚", "满心欢喜", "特别喜欢", "非常喜欢", "开心得忍不住笑"
]

NEUTRAL_PHRASES = [
    "快递状态更新", "正在派送", "没有明显情绪", "普通的信息更新",
    "按时间安排", "正常推进", "照常处理", "相关内容已经记录",
    "后续再按通知处理", "流程到这里结束"
]

GENERAL_NEGATIVE_PHRASES = [
    "真的很差", "不是很好", "不太好", "不好看", "不太行",
    "不满意", "不舒服", "感觉一般"
]


def make_phrase_terms():
    phrase_terms = set(
        ANGER_PHRASES
        + DISAPPOINTMENT_PHRASES
        + HAPPY_PHRASES
        + NEUTRAL_PHRASES
        + GENERAL_NEGATIVE_PHRASES
    )

    emotion_word_groups = [
        ANGER_WORDS,
        DISAPPOINTMENT_WORDS,
        HAPPY_WORDS,
        NEUTRAL_WORDS,
        GENERAL_NEGATIVE_WORDS,
    ]

    for degree_word in DEGREE_WORDS:
        for word_group in emotion_word_groups:
            for emotion_word in word_group:
                phrase_terms.add(degree_word + emotion_word)

    negative_focus_groups = [
        ANGER_WORDS,
        DISAPPOINTMENT_WORDS,
        GENERAL_NEGATIVE_WORDS,
    ]
    for negation_word in NEGATION_WORDS:
        for word_group in negative_focus_groups:
            for emotion_word in word_group:
                phrase_terms.add(negation_word + emotion_word)
        for degree_word in DEGREE_WORDS:
            for word_group in negative_focus_groups:
                for emotion_word in word_group:
                    phrase_terms.add(negation_word + degree_word + emotion_word)

    return sorted(phrase_terms, key=len, reverse=True)


PHRASE_TERMS = make_phrase_terms()
WORD_TERMS = sorted(
    set(
        GENERAL_WORDS
        + HAPPY_WORDS
        + ANGER_WORDS
        + DISAPPOINTMENT_WORDS
        + NEUTRAL_WORDS
        + GENERAL_NEGATIVE_WORDS
        + GENERAL_NEGATIVE_PHRASES
        + DEGREE_WORDS
        + NEGATION_WORDS
    ),
    key=len,
    reverse=True,
)


def longest_match_tokenize(text, terms):
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


def char_tokenize(text):
    return [ch for ch in text if ch not in PUNCTUATIONS]


def word_tokenize(text):
    return longest_match_tokenize(text, WORD_TERMS)


def phrase_tokenize(text):
    return longest_match_tokenize(text, PHRASE_TERMS)


def print_tokenization_examples():
    test_texts = [
        "这个电影不是很好，我有点失望",
        "客服态度太离谱了，我真的很生气",
        "今天收到礼物，真的非常开心",
        "快递状态更新为正在派送",
    ]

    print("\n====== 三路切分示例 ======")
    for text in test_texts:
        print(f"\n文本: {text}")
        print(f"char_tokenize: {char_tokenize(text)}")
        print(f"word_tokenize: {word_tokenize(text)}")
        print(f"phrase_tokenize: {phrase_tokenize(text)}")
