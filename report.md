# 基于循环神经网络的命名实体识别实验报告

## 1. 问题描述

命名实体识别是自然语言处理中的序列标注任务，目标是在给定句子中识别人名、组织机构、地点和其他专有名词等实体。本实验使用 CoNLL-2003 数据集，采用 BIO 标注格式，对 `PER`、`ORG`、`LOC`、`MISC` 四类实体进行识别。

## 2. 数据集

CoNLL-2003 数据集包含英文新闻文本，数据按句子组织，每个 token 对应一个命名实体标签。标签包括 `O`、`B-PER`、`I-PER`、`B-ORG`、`I-ORG`、`B-LOC`、`I-LOC`、`B-MISC`、`I-MISC`。其中 `B-` 表示实体开始，`I-` 表示实体内部，`O` 表示非实体。

## 3. 模型设计

本实验实现了基于循环神经网络的 BiLSTM 序列标注模型。模型包含三部分：

1. 词嵌入层：将输入 token ID 映射为稠密向量。
2. 双向 LSTM：同时建模句子左侧和右侧上下文信息。
3. 线性分类层：对每个 token 输出一个 BIO 标签。

训练时使用交叉熵损失函数，并通过 `ignore_index=-100` 忽略 padding 部分，避免填充 token 影响模型更新。优化器使用 AdamW，同时加入 dropout 和梯度裁剪以提升训练稳定性。

## 4. 评价指标

实验采用实体级 Precision、Recall 和 F1 作为主要评价指标。只有当预测实体的类别、起始位置和结束位置都与真实实体一致时，才认为该实体预测正确。

```text
Precision = 正确预测实体数 / 预测实体总数
Recall    = 正确预测实体数 / 真实实体总数
F1        = 2 * Precision * Recall / (Precision + Recall)
```

## 5. 实验步骤

安装依赖：

```bash
python3 -m pip install -r requirements.txt
```

运行训练：

```bash
python3 src/train_ner.py --epochs 8 --batch_size 32 --output_dir outputs/conll2003
```

离线自检：

```bash
python3 src/train_ner.py --use_sample --epochs 3 --batch_size 2 --hidden_dim 32 --embedding_dim 32 --output_dir outputs/sample
```

## 6. 实验结果

一次离线小样本自检结果可在 `outputs/sample/metrics.json` 查看。为了在当前环境中快速验证真实数据流程，本实验还使用 CoNLL-2003 的 1000 条训练样本和 500 条验证/测试样本运行了子集实验，命令如下：

```bash
python3 src/train_ner.py --epochs 20 --batch_size 32 --hidden_dim 64 --embedding_dim 64 --max_train_examples 1000 --max_eval_examples 500 --output_dir outputs/conll2003_subset
```

子集实验测试集结果如下：

| 模型 | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| BiLSTM | 0.5361 | 0.4608 | 0.4956 |

若使用完整训练集，可去掉 `--max_train_examples` 和 `--max_eval_examples` 参数并适当增大 `hidden_dim` 或训练轮数：

```bash
python3 src/train_ner.py --epochs 8 --batch_size 32 --output_dir outputs/conll2003
```

## 7. 分析

BiLSTM 能够利用前后文信息，比单向 RNN 更适合序列标注任务。例如在识别组织机构名或人名时，模型不仅可以利用当前词本身，还可以利用相邻词和句子结构信息。实验中实体级 F1 比 token accuracy 更重要，因为命名实体识别关注完整实体边界和实体类型是否同时正确。

模型的主要不足是只使用随机初始化词向量，没有引入预训练词向量或字符级特征，因此对于未登录词、罕见词和拼写变化较多的实体泛化能力有限。后续可以加入 CRF 层约束 BIO 标签转移，或使用预训练语言模型进一步提升效果。

## 8. 参考资料

- CoNLL-2003 shared task: https://www.clips.uantwerpen.be/conll2003/ner/
- Neural Architectures for Named Entity Recognition: https://arxiv.org/pdf/1603.01360.pdf
- End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF: https://arxiv.org/pdf/1603.01354.pdf
