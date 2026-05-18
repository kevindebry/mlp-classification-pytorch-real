# Chinese Emotion Classification with Multi-View TextCNN

本项目是一个基于 PyTorch 的中文微博情绪分类实验项目，目标是对中文短文本进行四分类情绪识别：

- 喜悦
- 愤怒
- 厌恶
- 低落

项目最初从基础文本分类实验出发，逐步扩展为一个包含数据预处理、三路文本视角建模、训练评估、误差分析、数据质量审计和消融实验的完整小型深度学习项目。

当前版本的核心模型是 **三分支 TextCNN + 门控融合机制**。模型分别从字粒度、词粒度和短语粒度提取文本特征，再通过 gate 对不同分支特征进行加权融合，最后完成情绪分类。

---

## 1. 项目特点

本项目不是单纯调用现成模型，而是围绕一个中文情绪分类任务，完整实现了从数据处理到模型分析的训练流程。

主要特点包括：

1. **中文微博情绪四分类**
   - 支持喜悦、愤怒、厌恶、低落四类情绪识别。
   - 数据格式统一为 `text,label`。

2. **多视角文本表示**
   - 字粒度 `char`
   - 词粒度 `word`
   - 短语粒度 `phrase`

3. **三分支 TextCNN**
   - 每个分支独立进行 Embedding、Conv1d、ReLU、Max Pooling。
   - 不同粒度使用不同最大长度、卷积核大小和 embedding 维度。

4. **门控融合机制**
   - 将 char、word、phrase 三路特征拼接后输入 gate。
   - gate 输出每一路特征的权重。
   - 通过加权后的多路特征进行最终分类。

5. **完整训练流程**
   - AdamW 优化器
   - CrossEntropyLoss
   - 类别权重 class weight
   - ReduceLROnPlateau 学习率调度
   - Early stopping
   - 保存最佳验证集 macro-F1 模型

6. **完整评估与诊断**
   - Accuracy
   - Macro Precision
   - Macro Recall
   - Macro F1
   - 每类 precision / recall / F1
   - 混淆矩阵
   - 错分样本导出
   - 重点错分类型分析
   - 低落类专项诊断

7. **数据质量审计**
   - 类别分布检查
   - 文本长度检查
   - 重复文本检查
   - 同文本多标签冲突检查
   - 关键词启发式标签异常检查
   - 模板化数据风险检查

8. **消融实验**
   - char only
   - word only
   - phrase only
   - char + word
   - char + word + phrase concat
   - char + word + phrase gated fusion

---

## 2. 项目结构

```text
mlp-classification-pytorch-real/
│
├── config.py
├── prepare_weibo_txt_dataset.py
├── data_quality_audit.py
├── data_utils.py
├── tokenizer.py
├── model.py
├── train.py
├── evaluate.py
├── predict.py
├── main.py
├── ablation.py
├── utils.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/
│   │   ├── 0_simplifyweibo.txt
│   │   ├── 1_simplifyweibo.txt
│   │   ├── 2_simplifyweibo.txt
│   │   └── 3_simplifyweibo.txt
│   │
│   ├── weibo_train.csv
│   ├── weibo_val.csv
│   └── weibo_test.csv
│
├── cache/
│   ├── tokenized_xxx.json
│   └── encoded_xxx.npy
│
├── output/
│   ├── best_three_branch_textcnn.pt
│   ├── multiview_vocabs.json
│   ├── label_map.json
│   ├── test_metrics_three_branch.json
│   ├── training_history_three_branch.json
│   ├── training_curves_three_branch.png
│   ├── confusion_matrix_three_branch.png
│   ├── error_samples_three_branch.csv
│   ├── error_analysis.csv
│   ├── error_summary.txt
│   ├── class_distribution_report_three_branch.json
│   ├── low_missed_errors_three_branch.csv
│   └── predicted_low_samples_three_branch.csv
│
└── data_quality_report/
    ├── dataset_basic_info.txt
    ├── label_distribution.csv
    ├── text_length_summary.csv
    ├── duplicated_texts.csv
    ├── duplicated_text_conflicting_labels.csv
    ├── top_repeated_texts.csv
    ├── possible_label_mismatch.csv
    ├── weak_label_evidence_samples.csv
    ├── label_sanity_summary.txt
    └── suspicious_summary.txt
```

---

## 3. 版本迭代过程

### v1.0：基础文本分类原型

早期版本主要目标是跑通中文文本分类的最小流程：

```text
CSV 数据读取
→ 文本清洗
→ 标签编码
→ 模型训练
→ 测试集评估
```

这一阶段的重点不是追求模型复杂度，而是先验证：

- 数据能否正常读取；
- 标签是否能正确映射；
- PyTorch 训练流程是否能跑通；
- 模型是否能在真实中文文本上产生可用结果。

该阶段更接近一个基础分类 baseline，为后续 CNN、多粒度文本表示和误差分析打基础。

### v1.1：微博情绪数据集整理

在 v1.1 中，项目加入了微博情绪数据预处理脚本 `prepare_weibo_txt_dataset.py`。

该版本完成了从原始 txt 文件到标准 CSV 数据集的转换。原始数据按类别存放：

```text
data/raw/0_simplifyweibo.txt    # 喜悦
data/raw/1_simplifyweibo.txt    # 愤怒
data/raw/2_simplifyweibo.txt    # 厌恶
data/raw/3_simplifyweibo.txt    # 低落
```

处理流程包括：

1. 读取原始 txt 文本；
2. 去除词性标注；
3. 清洗空文本和过短文本；
4. 去重；
5. 按类别平衡采样；
6. 划分 train / val / test；
7. 保存为标准 CSV 文件。

输出文件：

```text
data/weibo_train.csv
data/weibo_val.csv
data/weibo_test.csv
```

每条样本格式为：

```csv
text,label
今天真的很开心,0
气死我了,1
太恶心了,2
有点难过,3
```

### v1.2：多粒度文本切分

v1.2 开始不再只把文本当作单一序列处理，而是加入了三种文本视角：

| 视角 | 含义 | 作用 |
| --- | --- | --- |
| char | 字粒度 | 保留中文短文本中的细粒度信息 |
| word | 词粒度 | 捕捉常见情绪词和语义单元 |
| phrase | 短语粒度 | 捕捉“非常开心”“不太舒服”“真的很失望”等组合表达 |

对应代码主要在 `tokenizer.py` 中实现。该版本加入了：

- 情绪词表；
- 程度副词词表；
- 否定词词表；
- 情绪短语词表；
- longest-match 最长匹配分词；
- Trie 加速版本；
- tokenizer 一致性测试。

三种 tokenizer 分别为：

```python
char_tokenize(text)
word_tokenize(text)
phrase_tokenize(text)
```

这一阶段的核心意义是：模型不再只依赖单一文本粒度，而是尝试从不同层级捕捉中文情绪表达。

### v1.3：三分支 TextCNN 模型

v1.3 将模型升级为三分支 TextCNN。每个分支结构如下：

```text
input_ids
→ Embedding
→ Conv1d
→ ReLU
→ Max Pooling
→ Dropout
→ branch feature
```

三个分支分别处理：

```text
char_ids
word_ids
phrase_ids
```

模型结构大致为：

```text
char branch   ┐
word branch   ├─ feature fusion → classifier → logits
phrase branch ┘
```

核心模块：

```python
CNNBranch
AblationEmotionCNN
MultiViewEmotionCNN
```

其中 `CNNBranch` 是基础 CNN 特征提取模块，`MultiViewEmotionCNN` 是当前主模型。

这一版本的意义是：

- 字粒度适合处理中文短文本中的局部模式；
- 词粒度适合捕捉明确情绪词；
- 短语粒度适合捕捉否定、程度、组合语义；
- 三路结构比单一路径更适合做可解释的文本分类实验。

### v1.4：门控融合机制

v1.4 在三分支 TextCNN 的基础上加入 gate 机制。

普通 concat 融合方式是：

```text
char_feature + word_feature + phrase_feature
→ concat
→ classifier
```

门控融合方式是：

```text
char_feature
word_feature
phrase_feature
→ concat 得到 raw_fused_feature
→ gate 网络输出每个分支的权重
→ 每个分支特征乘以对应 gate weight
→ 再 concat
→ classifier
```

gate 的作用是让模型自己学习：

- 当前样本更依赖字粒度；
- 当前样本更依赖词粒度；
- 当前样本更依赖短语粒度。

例如：

```text
“哈哈哈今天太开心了”
```

可能更依赖 char 和 phrase。

```text
“恶心”“离谱”“失望”
```

可能更依赖 word。

```text
“不太开心”“没有很难过”
```

可能更依赖 phrase，因为否定词和程度副词组合会影响情绪判断。

当前模型使用：

```python
MultiViewEmotionCNN
```

默认启用：

```python
enabled_branches = ["char", "word", "phrase"]
use_gate = True
```

### v1.5：训练流程工程化

v1.5 对训练流程进行了工程化整理，主入口为 `main.py`。

完整训练流程：

```text
读取 train / val / test
→ 清洗数据
→ 打印类别分布
→ tokenizer 示例检查
→ Trie tokenizer 一致性测试
→ 三路 tokenizer
→ 构建三路 vocab
→ 编码为 id 序列
→ 构建 DataLoader
→ 初始化三分支 TextCNN
→ 计算类别权重
→ 训练
→ 验证集评估
→ 保存最佳模型
→ 测试集评估
→ 保存输出结果
```

训练配置集中放在 `config.py` 中，包括：

```python
EPOCHS = 50
BATCH_SIZE = 64
LR = 0.0008
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 1.0
PATIENCE = 6
SAVE_METRIC = "val_macro_f1"
USE_CLASS_WEIGHT = True
```

优化策略：

- AdamW
- ReduceLROnPlateau
- gradient clipping
- early stopping
- 按验证集 macro-F1 保存最佳模型

这一版本的意义是：项目不再只是“能训练”，而是形成了较完整、可复现、可分析的训练管线。

### v1.6：评估、可视化与误差分析

v1.6 加入了更完整的评估模块。测试阶段会输出：

- loss
- accuracy
- macro precision
- macro recall
- macro F1
- 每个类别的 precision / recall / F1
- confusion matrix

同时保存：

```text
output/test_metrics_three_branch.json
output/training_history_three_branch.json
output/training_curves_three_branch.png
output/confusion_matrix_three_branch.png
```

此外，项目会导出错分样本：

```text
output/error_samples_three_branch.csv
output/error_analysis.csv
output/error_summary.txt
```

并额外关注几类重要错分：

```text
喜悦 -> 低落
愤怒 -> 低落
厌恶 -> 低落
愤怒 -> 厌恶
厌恶 -> 愤怒
```

对应输出：

```text
output/joy_to_sadness.csv
output/anger_to_sadness.csv
output/disgust_to_sadness.csv
output/anger_to_disgust.csv
output/disgust_to_anger.csv
```

由于“低落”类在情绪分类中容易和其他负向情绪混淆，项目还单独导出：

```text
output/low_missed_errors_three_branch.csv
output/predicted_low_samples_three_branch.csv
```

这一版本的重点是：不仅看总体准确率，还要分析模型到底错在哪里。

### v1.7：数据质量审计

v1.7 加入了 `data_quality_audit.py`，用于检查数据本身是否存在问题。

该脚本会自动检查：

1. 数据基本信息；
2. 类别分布；
3. 文本长度；
4. 空文本和过短文本；
5. 重复文本；
6. 同文本不同标签冲突；
7. 情绪关键词与标签是否疑似不一致；
8. 数据是否存在模板化或过度采样痕迹。

输出目录：

```text
data_quality_report/
```

主要报告：

```text
dataset_basic_info.txt
label_distribution.csv
text_length_summary.csv
duplicated_texts.csv
duplicated_text_conflicting_labels.csv
top_repeated_texts.csv
possible_label_mismatch.csv
weak_label_evidence_samples.csv
label_sanity_summary.txt
suspicious_summary.txt
```

这一版本的意义是：如果模型效果不好，不一定是模型结构问题，也可能是数据本身存在噪声、重复、标签冲突或类别分布问题。因此需要先审计数据，再解释模型表现。

### v1.8：消融实验

v1.8 加入了 `ablation.py`，用于比较不同分支和不同融合方式的效果。

实验配置包括：

```python
experiments = [
    {"name": "char_only", "enabled_branches": ["char"], "use_gate": False},
    {"name": "word_only", "enabled_branches": ["word"], "use_gate": False},
    {"name": "phrase_only", "enabled_branches": ["phrase"], "use_gate": False},
    {"name": "char_word_concat", "enabled_branches": ["char", "word"], "use_gate": False},
    {"name": "char_word_phrase_concat", "enabled_branches": ["char", "word", "phrase"], "use_gate": False},
    {"name": "char_word_phrase_gated", "enabled_branches": ["char", "word", "phrase"], "use_gate": True},
]
```

输出文件：

```text
ablation_results_weibo.csv
```

记录指标：

- test loss
- test accuracy
- macro precision
- macro recall
- macro F1

该版本用于回答：

1. 字粒度是否有效？
2. 词粒度是否有效？
3. 短语粒度是否有效？
4. 多分支是否优于单分支？
5. gate 是否优于普通 concat？
6. 模型提升来自结构设计，还是只是参数量增加？

---

## 4. 安装依赖

建议使用 Python 3.10 或以上版本。

```bash
pip install -r requirements.txt
```

也可以手动安装核心依赖：

```bash
pip install torch pandas numpy matplotlib scikit-learn tqdm
```

其中 `scikit-learn` 主要用于：

- train / val / test 数据划分；
- balanced class weight 计算。

---

## 5. 数据准备

将原始微博情绪数据放入：

```text
data/raw/
```

文件命名：

```text
0_simplifyweibo.txt
1_simplifyweibo.txt
2_simplifyweibo.txt
3_simplifyweibo.txt
```

类别对应关系：

| label | emotion |
| --- | --- |
| 0 | 喜悦 |
| 1 | 愤怒 |
| 2 | 厌恶 |
| 3 | 低落 |

运行数据预处理：

```bash
python prepare_weibo_txt_dataset.py
```

生成：

```text
data/weibo_train.csv
data/weibo_val.csv
data/weibo_test.csv
```

---

## 6. 数据质量审计

训练前建议先运行：

```bash
python data_quality_audit.py
```

如果使用默认路径，脚本会读取：

```text
data/weibo_train.csv
data/weibo_val.csv
data/weibo_test.csv
```

也可以指定单个文件：

```bash
python data_quality_audit.py --input data/weibo_train.csv
```

或指定三份数据：

```bash
python data_quality_audit.py \
  --train data/weibo_train.csv \
  --val data/weibo_val.csv \
  --test data/weibo_test.csv
```

审计结果会保存到：

```text
data_quality_report/
```

---

## 7. 训练模型

运行主训练流程：

```bash
python main.py
```

训练过程中会自动完成：

1. 数据读取；
2. 文本清洗；
3. 三路 tokenizer；
4. 构建 vocab；
5. token id 编码；
6. 构建 DataLoader；
7. 模型训练；
8. 验证集评估；
9. 保存最佳模型；
10. 测试集评估；
11. 输出图表和错误分析文件。

---

## 8. 消融实验

运行：

```bash
python ablation.py
```

输出：

```text
ablation_results_weibo.csv
```

该文件用于比较不同模型结构的效果。

---

## 9. 单句预测

`main.py` 在交互式终端中运行结束后，会进入单句预测模式：

```text
输入一句中文文本预测情绪，直接回车退出。
请输入文本:
```

示例：

```text
请输入文本: 今天真的很开心
预测类别: 喜悦 (0)
```

---

## 10. 当前模型结构说明

当前主模型为：

```text
ThreeBranchTextCNN-SigmoidGatedFusion
```

整体结构：

```text
text
│
├── char tokenizer   → char ids   → char CNN branch
├── word tokenizer   → word ids   → word CNN branch
└── phrase tokenizer → phrase ids → phrase CNN branch
                              │
                              ↓
                    gated feature fusion
                              │
                              ↓
                         classifier
                              │
                              ↓
                           logits
```

每个 CNN branch：

```text
Embedding
→ Conv1d with multiple kernel sizes
→ ReLU
→ Max Pooling over time
→ Dropout
```

融合层：

```text
concat(char_feature, word_feature, phrase_feature)
→ gate network
→ sigmoid gate weights
→ gated branch features
→ concat
→ classifier
```

分类器：

```text
Linear
→ ReLU
→ Dropout
→ Linear
→ logits
```

---

## 11. 输出文件说明

训练完成后，主要输出如下：

| 文件 | 说明 |
| --- | --- |
| `output/best_three_branch_textcnn.pt` | 最佳模型 checkpoint |
| `output/multiview_vocabs.json` | char / word / phrase 三路词表 |
| `output/label_map.json` | 标签映射 |
| `output/test_metrics_three_branch.json` | 测试集指标 |
| `output/training_history_three_branch.json` | 训练历史 |
| `output/training_curves_three_branch.png` | 训练曲线 |
| `output/confusion_matrix_three_branch.png` | 混淆矩阵 |
| `output/error_samples_three_branch.csv` | 错分样本 |
| `output/error_analysis.csv` | 带概率的错分分析 |
| `output/error_summary.txt` | 错误类型统计 |
| `output/class_distribution_report_three_branch.json` | 类别诊断报告 |
| `output/low_missed_errors_three_branch.csv` | 低落类漏判样本 |
| `output/predicted_low_samples_three_branch.csv` | 被预测为低落的样本 |

---

## 12. 项目反思

这个项目的重点不只是训练出一个分类器，而是通过一个完整任务理解深度学习项目的基本工程流程：

```text
数据质量
→ 文本表示
→ 模型结构
→ 训练策略
→ 评估指标
→ 错误分析
→ 消融实验
```

相比只看 accuracy，本项目更关注：

- 哪些类别更难分；
- 模型是否偏向某些类别；
- “低落”和其他负向情绪为什么容易混淆；
- 多粒度输入是否真的带来提升；
- gate 是否比普通 concat 更有效；
- 数据标签本身是否存在噪声。

因此，这个项目更适合作为一个本科阶段的深度学习/NLP 工程练习项目，而不是单纯追求最高分数的竞赛项目。

---

## 13. 后续改进方向

后续可以继续从以下方向改进：

1. **替换更强的文本编码器**
   - 使用 BiLSTM、Transformer Encoder 或 BERT 类模型进行对比。

2. **改进 tokenizer**
   - 当前 word / phrase tokenizer 主要基于规则词表和最长匹配，可以尝试引入 jieba 或训练子词 tokenizer。

3. **增强可解释性**
   - 输出 gate 权重；
   - 分析不同情绪样本中 char / word / phrase 分支的重要性。

4. **改进数据集质量**
   - 人工检查高置信错分样本；
   - 清理疑似错标样本；
   - 减少重复文本和模板化文本。

5. **加入实验报告**
   - 将 ablation 结果整理成表格；
   - 对混淆矩阵进行文字分析；
   - 对不同版本的模型效果进行复盘。

6. **整理为科研展示版本**
   - 补充项目动机；
   - 补充模型结构图；
   - 补充训练曲线；
   - 补充错误分析案例；
   - 形成一份完整的项目展示文档。

---

## 14. 一句话总结

本项目从基础中文文本分类出发，逐步迭代为一个包含多粒度文本建模、三分支 TextCNN、门控融合、完整训练评估、数据质量审计和消融实验的中文微博情绪分类系统。
