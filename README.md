# 基于循环神经网络的命名实体识别

本项目使用 PyTorch 实现 BiLSTM 序列标注模型，在 CoNLL-2003 命名实体识别数据集上识别 `PER`、`ORG`、`LOC`、`MISC` 四类实体，并输出实体级 Precision、Recall、F1。

## 环境

```bash
python3 -m pip install -r requirements.txt
```

## 快速自检

离线环境可以先运行内置小样本，确认训练、评估和指标计算流程可用：

```bash
python3 src/train_ner.py --use_sample --epochs 3 --batch_size 2 --hidden_dim 32 --embedding_dim 32 --output_dir outputs/sample
```

## 使用 CoNLL-2003

如果当前环境可以访问 Hugging Face Datasets，直接运行：

```bash
python3 src/train_ner.py --epochs 8 --batch_size 32 --output_dir outputs/conll2003
```

也可以手动下载 CoNLL 格式数据，并按如下文件名放入目录：

```text
data/conll2003/train.txt
data/conll2003/dev.txt
data/conll2003/test.txt
```

然后运行：

```bash
python3 src/train_ner.py --data_dir data/conll2003 --epochs 8 --batch_size 32 --output_dir outputs/conll2003
```

脚本会保存：

- `best_model.pt`：验证集 F1 最好的模型参数
- `metrics.json`：训练历史和测试集 Precision、Recall、F1
- `vocab.json`、`label_to_id.json`：词表和标签映射

## 方法

模型结构为词嵌入层、双向 LSTM 和线性分类层。输入句子先转换为词 ID，经过 embedding 得到词向量，再由 BiLSTM 编码上下文信息，最后对每个 token 输出 BIO 标签。损失函数使用交叉熵，并忽略 padding 位置。

评价采用实体级匹配：预测实体的类别、起始位置和结束位置都与真实实体一致时才计为正确。指标包括：

```text
Precision = 正确预测实体数 / 预测实体总数
Recall    = 正确预测实体数 / 真实实体总数
F1        = 2 * Precision * Recall / (Precision + Recall)
```
