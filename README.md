# 基于循环神经网络的命名实体识别实验报告

## 1. 实验目的

本实验旨在使用 PyTorch 实现基于循环神经网络的英文命名实体识别（Named Entity Recognition, NER）模型，在 CoNLL-2003 数据集上识别人名、组织机构、地点和其他专有名词四类实体，并使用实体级 Precision、Recall 和 F1 评价模型效果。

本实验的具体目标如下：

1. 掌握命名实体识别任务的基本流程；
2. 熟悉 BIO 序列标注格式、数据加载、词表构建和实体级评价方法；
3. 实现并对比 BiLSTM-softmax、Char-CNN + BiLSTM + CRF 和加入 GloVe 词向量后的模型；
4. 参考 Ma & Hovy (2016) 与 Lample et al. (2016) 的结构设计，分析当前实现与论文结果之间的差距。

---

## 2. 实验原理

### 2.1 命名实体识别任务简介

命名实体识别是自然语言处理中的序列标注任务。给定一个 token 序列：

```text
EU rejects German call to boycott British lamb .
```

模型需要为每个 token 输出一个 BIO 标签：

```text
B-ORG O B-MISC O O O B-MISC O O
```

其中 `B-` 表示实体开始，`I-` 表示实体内部，`O` 表示非实体。本实验识别 `PER`、`ORG`、`LOC`、`MISC` 四类实体。

### 2.2 BiLSTM 序列编码

BiLSTM 同时从左到右和从右到左编码句子上下文，使每个 token 的表示能够包含前后文信息。对于 NER 任务，这比普通单向 RNN 更适合处理实体边界和实体类型判断。

初始实现使用：

```text
word embedding -> BiLSTM -> Linear -> softmax
```

该结构能完成基本序列标注，但每个位置独立分类，没有显式建模 BIO 标签转移关系。

### 2.3 Char-CNN + BiLSTM + CRF

为了对齐参考论文，本实验进一步实现了：

```text
word embedding + character CNN -> BiLSTM -> CRF
```

其中：

1. 词嵌入层将 token ID 映射为稠密向量；
2. 字符 CNN 从单词字符序列中提取大小写、后缀、形态等词内部特征；
3. BiLSTM 编码上下文信息；
4. CRF 建模标签之间的转移关系，并输出全局最优标签序列。

CRF 层可以学习 `B-ORG` 后更可能接 `I-ORG`，`I-PER` 不应随意接在 `B-LOC` 后等序列约束，因此比逐 token softmax 更适合 BIO 标注。

### 2.4 预训练词向量与训练策略

参考论文通常使用无监督预训练词向量增强泛化能力。本实验下载并使用 GloVe 6B 词向量，主要采用 `glove.6B.100d.txt`。由于 GloVe 6B 词表本身为小写，本实验在最佳配置中加入 `--lower`，同时保留字符 CNN 来捕捉大小写信息。

训练中还加入：

- dropout；
- singleton word dropout；
- ReduceLROnPlateau 学习率衰减；
- early stopping；
- 梯度裁剪。

---

## 3. 数据集与评价指标

### 3.1 CoNLL-2003 数据集

CoNLL-2003 是英文命名实体识别常用数据集，文本主要来自新闻语料。数据按句子组织，每个 token 对应一个 NER 标签。

本实验最终采用 Hugging Face `datasets` 加载 `conll2003` 数据集，训练、验证和测试划分均来自该数据源。仓库中保留的 `data/conll2003` 仅作为离线缓存和备用导出目录，不作为最终实验的主要入口。

### 3.2 评价指标

实验采用实体级 Precision、Recall 和 F1。只有当预测实体的类别、起始位置和结束位置都与真实实体一致时，才认为该实体预测正确。

```text
Precision = 正确预测实体数 / 预测实体总数
Recall    = 正确预测实体数 / 真实实体总数
F1        = 2 * Precision * Recall / (Precision + Recall)
```

另外程序也输出 token accuracy，但最终以实体级 F1 为主要指标。

---

## 4. 实验环境与参数设置

### 4.1 实验环境

```bash
python3 -m pip install -r requirements.txt
```

主要依赖如下：

| 依赖 | 用途 |
| --- | --- |
| PyTorch | 模型训练与推理 |
| datasets | 加载 Hugging Face CoNLL-2003 数据集 |
| NumPy | 随机种子与数值处理 |
| tqdm | 训练进度显示 |

### 4.2 默认模型参数

| 参数 | 取值 |
| --- | ---: |
| 模型 | Char-CNN + BiLSTM + CRF |
| Word Embedding | 128（随机初始化时） |
| Hidden Dim | 256 |
| LSTM Layers | 1 |
| Char Embedding Dim | 30 |
| Char CNN Channels | 30 |
| Dropout | 0.3 |
| Optimizer | AdamW |
| Learning Rate | 0.005 |
| Batch Size | 32 |

### 4.3 最佳 GloVe 配置

| 参数 | 取值 |
| --- | ---: |
| 预训练词向量 | GloVe 6B 100d |
| Lowercase | true |
| Dropout | 0.5 |
| Singleton Word Dropout | 0.05 |
| Learning Rate | 0.003 |
| LR Decay Patience | 3 |
| LR Decay Factor | 0.5 |
| Early Stop Patience | 8 |
| Best Epoch | 6 |

---

## 5. 项目目录结构说明

当前项目核心目录如下：

```text
|-- README.md              # 项目说明与实验报告
|-- requirements.txt       # Python 依赖
|-- src
|   `-- train_ner.py       # 数据加载、模型定义、训练与评估主脚本
|-- data                   # datasets 导出的离线缓存目录，已加入 .gitignore
|-- embeddings             # GloVe 词向量目录，已加入 .gitignore
`-- outputs                # 训练输出、模型权重和指标文件，已加入 .gitignore
```

最终保留的 `outputs` 结果目录如下：

```text
outputs/conll2003_char_crf_full          # 随机初始化 Char-CNN + BiLSTM + CRF
outputs/conll2003_glove                  # 直接加入 GloVe 100d
outputs/conll2003_glove100_lower_tuned   # 最终最佳配置
```

每个正式输出目录中保存：

- `best_model.pt`：验证集 F1 最好的模型参数；
- `metrics.json`：训练历史、最佳验证集 F1 和测试集指标；
- `vocab.json`：词表；
- `char_vocab.json`：字符表；
- `label_to_id.json`：标签映射。

---

## 6. 实验流程

### 6.1 基础模型训练

基础实验使用 `datasets` 加载 CoNLL-2003，并训练随机初始化词向量的 Char-CNN + BiLSTM + CRF 模型：

```bash
python3 src/train_ner.py \
  --dataset_name conll2003 \
  --epochs 8 \
  --batch_size 32 \
  --output_dir outputs/conll2003_char_crf_full
```

### 6.2 加入 GloVe 词向量

```bash
python3 src/train_ner.py \
  --dataset_name conll2003 \
  --epochs 20 \
  --batch_size 32 \
  --pretrained_embeddings embeddings/glove.6B.100d.txt \
  --output_dir outputs/conll2003_glove
```

### 6.3 最佳调参配置

当前最佳复现实验命令如下：

```bash
python3 src/train_ner.py \
  --dataset_name conll2003 \
  --epochs 30 \
  --batch_size 32 \
  --pretrained_embeddings embeddings/glove.6B.100d.txt \
  --lower \
  --dropout 0.5 \
  --word_dropout 0.05 \
  --lr 0.003 \
  --lr_decay_patience 3 \
  --lr_decay_factor 0.5 \
  --early_stop_patience 8 \
  --no_progress \
  --output_dir outputs/conll2003_glove100_lower_tuned
```

### 6.4 消融实验

关闭字符特征和 CRF，退化到普通 BiLSTM-softmax：

```bash
python3 src/train_ner.py \
  --dataset_name conll2003 \
  --no_char \
  --no_crf \
  --epochs 8 \
  --batch_size 32 \
  --output_dir outputs/bilstm_softmax
```

---

## 7. 实验结果

### 7.1 测试集结果对比

| 模型 | Precision | Recall | F1 | Token Acc |
| --- | ---: | ---: | ---: | ---: |
| BiLSTM-softmax（初始实现） | 0.6477 | 0.7525 | 0.6962 | 0.9289 |
| Char-CNN + BiLSTM + CRF | 0.8060 | 0.8304 | 0.8180 | 0.9612 |
| Char-CNN + BiLSTM + CRF + GloVe 100d | 0.8390 | 0.8341 | 0.8365 | 0.9676 |
| Char-CNN + BiLSTM + CRF + GloVe 100d + 调参 | **0.8687** | **0.8624** | **0.8656** | **0.9725** |

### 7.2 验证集最佳结果

最佳配置在验证集上达到：

```text
best_epoch = 6
valid_f1   = 0.9210
```

对应测试集结果为：

```text
precision = 0.8687
recall    = 0.8624
f1        = 0.8656
token_acc = 0.9725
```

### 7.3 与参考论文对比

| 方法 | CoNLL-2003 F1 |
| --- | ---: |
| Lample et al. (2016) BiLSTM-CRF + character representation + pretrained embeddings | 90+ |
| Ma & Hovy (2016) BiLSTM-CNNs-CRF | 91.21 |
| 本实验最佳结果 | 86.56 |

当前实现已经对齐了参考论文的主要结构：字符级表示、BiLSTM、CRF 和预训练词向量。但测试集 F1 仍低于论文结果，说明在词向量来源、训练细节、超参数搜索、随机种子稳定性以及官方评测脚本对齐方面仍有改进空间。

---

## 8. 结果分析

### 8.1 Char-CNN 与 CRF 的有效性

从初始 BiLSTM-softmax 到 Char-CNN + BiLSTM + CRF，测试集 F1 从 0.6962 提升到 0.8180，提升约 12.18 个百分点。主要原因是：

1. 字符 CNN 能捕获大小写、后缀和词形信息，对未登录词和罕见实体更友好；
2. CRF 能显式建模 BIO 标签转移，减少非法或不合理标签序列；
3. BiLSTM 提供上下文表示，适合判断实体边界和类型。

### 8.2 GloVe 的影响

直接加入 GloVe 100d 后，测试集 F1 从 0.8180 提升到 0.8365。继续加入 `--lower`、更强 dropout、singleton word dropout、学习率衰减和 early stopping 后，测试集 F1 进一步提升到 0.8656。

`--lower` 对本实验很关键，因为 GloVe 6B 是小写词表。lowercase 后词表规模从约 23625 降到约 21011，预训练向量覆盖更集中；字符 CNN 仍保留原始 token 的字符信息，因此大小写特征没有完全丢失。

### 8.3 不足与改进方向

当前最佳测试集 F1 为 86.56，验证集最佳 F1 为 92.10，二者仍存在约 5.5 个百分点差距，说明泛化能力仍有提升空间。后续可以尝试：

1. 多随机种子训练并报告平均值；
2. 使用官方 `conlleval` 脚本复核指标；
3. 尝试 SENNA、fastText 等其他预训练词向量；
4. 将 Char-CNN 替换或对比为 Char-BiLSTM；
5. 对 dropout、学习率、word dropout、batch size 做更系统的搜索；
6. 加入合法 BIO 转移约束，进一步限制 CRF 解码空间。

---

## 9. 结论

本实验基于 PyTorch 完成了 CoNLL-2003 英文命名实体识别系统，实现了从数据读取、词表构建、模型训练、验证集选择最佳模型到测试集评估的完整流程。

实验表明，初始 BiLSTM-softmax 虽然能完成基本序列标注，但实体级 F1 较低。加入字符 CNN 和 CRF 后，模型性能明显提升；进一步引入 GloVe 100d、lowercase、dropout、singleton word dropout、学习率衰减和 early stopping 后，测试集 F1 达到 **0.8656**。

与 Ma & Hovy (2016) 和 Lample et al. (2016) 的 90+ F1 相比，本实验仍有差距，但模型结构已经对齐参考论文的核心思路，实验结果也验证了字符级特征、CRF 解码和预训练词向量对 NER 任务的有效性。

---

## 10. 其他
项目已开源 https://github.com/ChriCheng/RNNRecon

---
## 11. 参考文献

1. Ma X, Hovy E. End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF. ACL, 2016. https://arxiv.org/pdf/1603.01354.pdf
2. Lample G, Ballesteros M, Subramanian S, et al. Neural Architectures for Named Entity Recognition. NAACL, 2016. https://arxiv.org/pdf/1603.01360.pdf
3. Tjong Kim Sang E F, De Meulder F. Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition. CoNLL, 2003.
4. Pennington J, Socher R, Manning C D. GloVe: Global Vectors for Word Representation. EMNLP, 2014.
