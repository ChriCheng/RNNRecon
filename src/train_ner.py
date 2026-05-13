import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_LABEL_ID = -100


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_token(token: str, lower: bool) -> str:
    return token.lower() if lower else token


def sample_data():
    train = [
        (["EU", "rejects", "German", "call", "to", "boycott", "British", "lamb", "."],
         ["B-ORG", "O", "B-MISC", "O", "O", "O", "B-MISC", "O", "O"]),
        (["Peter", "Blackburn"], ["B-PER", "I-PER"]),
        (["BRUSSELS", "1996-08-22"], ["B-LOC", "O"]),
        (["The", "European", "Commission", "said", "on", "Thursday", "."],
         ["O", "B-ORG", "I-ORG", "O", "O", "O", "O"]),
        (["Germany", "'s", "representative", "to", "the", "European", "Union", "."],
         ["B-LOC", "O", "O", "O", "O", "B-ORG", "I-ORG", "O"]),
        (["British", "officials", "met", "in", "London", "."],
         ["B-MISC", "O", "O", "O", "B-LOC", "O"]),
    ]
    valid = [
        (["France", "supports", "EU", "talks", "."], ["B-LOC", "O", "B-ORG", "O", "O"]),
        (["John", "Smith", "visited", "Berlin", "."], ["B-PER", "I-PER", "O", "B-LOC", "O"]),
    ]
    test = [
        (["British", "minister", "met", "Peter", "in", "Brussels", "."],
         ["B-MISC", "O", "O", "B-PER", "O", "B-LOC", "O"]),
        (["The", "European", "Commission", "rejected", "the", "proposal", "."],
         ["O", "B-ORG", "I-ORG", "O", "O", "O", "O"]),
    ]
    label_names = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
    return {"train": train, "validation": valid, "test": test}, label_names


def read_conll_file(path: Path):
    examples = []
    tokens = []
    labels = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    examples.append((tokens, labels))
                    tokens, labels = [], []
                continue
            if line.startswith("-DOCSTART-"):
                continue
            parts = line.split()
            tokens.append(parts[0])
            labels.append(parts[-1])
    if tokens:
        examples.append((tokens, labels))
    return examples


def load_from_conll_dir(data_dir: Path):
    candidates = {
        "train": ["train.txt", "train.conll", "eng.train"],
        "validation": ["valid.txt", "dev.txt", "eng.testa"],
        "test": ["test.txt", "eng.testb"],
    }
    splits = {}
    for split, names in candidates.items():
        found = next((data_dir / name for name in names if (data_dir / name).exists()), None)
        if found is None:
            raise FileNotFoundError(f"Cannot find {split} file in {data_dir}. Tried: {', '.join(names)}")
        splits[split] = read_conll_file(found)

    labels = sorted({label for rows in splits.values() for _, labs in rows for label in labs})
    if "O" in labels:
        labels.remove("O")
        labels = ["O"] + labels
    return splits, labels


def load_from_huggingface():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets or pass --data_dir/--use_sample.") from exc

    dataset = load_dataset("conll2003", trust_remote_code=True)
    label_names = dataset["train"].features["ner_tags"].feature.names
    splits = {}
    for split in ["train", "validation", "test"]:
        split_rows = []
        for row in dataset[split]:
            split_rows.append((row["tokens"], [label_names[i] for i in row["ner_tags"]]))
        splits[split] = split_rows
    return splits, list(label_names)


def load_data(args):
    if args.use_sample:
        return sample_data()
    if args.data_dir:
        return load_from_conll_dir(Path(args.data_dir))
    return load_from_huggingface()


def build_vocab(examples, lower: bool, min_freq: int):
    counter = Counter()
    for tokens, _ in examples:
        counter.update(normalize_token(token, lower) for token in tokens)

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, freq in counter.most_common():
        if freq >= min_freq and token not in vocab:
            vocab[token] = len(vocab)
    return vocab


class NERDataset(Dataset):
    def __init__(self, examples, vocab, label_to_id, lower: bool, max_examples: int = 0):
        if max_examples:
            examples = examples[:max_examples]
        self.rows = []
        for tokens, labels in examples:
            token_ids = [vocab.get(normalize_token(token, lower), vocab[UNK_TOKEN]) for token in tokens]
            label_ids = [label_to_id[label] for label in labels]
            self.rows.append((tokens, token_ids, label_ids))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def collate_batch(batch):
    tokens, token_ids, label_ids = zip(*batch)
    lengths = torch.tensor([len(ids) for ids in token_ids], dtype=torch.long)
    max_len = max(lengths).item()

    input_ids = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), PAD_LABEL_ID, dtype=torch.long)
    mask = torch.zeros((len(batch), max_len), dtype=torch.bool)

    for i, (ids, labs) in enumerate(zip(token_ids, label_ids)):
        size = len(ids)
        input_ids[i, :size] = torch.tensor(ids, dtype=torch.long)
        labels[i, :size] = torch.tensor(labs, dtype=torch.long)
        mask[i, :size] = True
    return {"tokens": tokens, "input_ids": input_ids, "labels": labels, "lengths": lengths, "mask": mask}


class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, num_labels, embedding_dim, hidden_dim, num_layers, dropout, pad_id):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, input_ids, lengths):
        embeddings = self.dropout(self.embedding(input_ids))
        packed = nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        return self.classifier(self.dropout(output))


def split_label(label: str):
    if label == "O":
        return "O", ""
    if "-" not in label:
        return label, ""
    prefix, entity_type = label.split("-", 1)
    return prefix, entity_type


def extract_entities(labels):
    entities = []
    entity_type = None
    start = None

    for idx, label in enumerate(labels + ["O"]):
        prefix, current_type = split_label(label)
        starts_new = prefix == "B" or (prefix == "I" and (entity_type is None or current_type != entity_type))
        closes_current = prefix == "O" or starts_new

        if closes_current and entity_type is not None:
            entities.append((entity_type, start, idx - 1))
            entity_type = None
            start = None
        if starts_new:
            entity_type = current_type
            start = idx
    return set(entities)


def compute_metrics(gold_sequences, pred_sequences):
    true_positive = 0
    predicted_total = 0
    gold_total = 0
    token_correct = 0
    token_total = 0

    for gold, pred in zip(gold_sequences, pred_sequences):
        gold_entities = extract_entities(gold)
        pred_entities = extract_entities(pred)
        true_positive += len(gold_entities & pred_entities)
        predicted_total += len(pred_entities)
        gold_total += len(gold_entities)
        token_correct += sum(g == p for g, p in zip(gold, pred))
        token_total += len(gold)

    precision = true_positive / predicted_total if predicted_total else 0.0
    recall = true_positive / gold_total if gold_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = token_correct / token_total if token_total else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "token_accuracy": accuracy}


@torch.no_grad()
def evaluate(model, dataloader, id_to_label, device):
    model.eval()
    gold_sequences = []
    pred_sequences = []
    total_loss = 0.0
    total_batches = 0
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_LABEL_ID)

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        lengths = batch["lengths"].to(device)
        logits = model(input_ids, lengths)
        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        total_loss += loss.item()
        total_batches += 1

        predictions = logits.argmax(dim=-1).cpu()
        for pred_row, gold_row in zip(predictions, batch["labels"]):
            gold = []
            pred = []
            for pred_id, gold_id in zip(pred_row.tolist(), gold_row.tolist()):
                if gold_id == PAD_LABEL_ID:
                    continue
                gold.append(id_to_label[gold_id])
                pred.append(id_to_label[pred_id])
            gold_sequences.append(gold)
            pred_sequences.append(pred)

    metrics = compute_metrics(gold_sequences, pred_sequences)
    metrics["loss"] = total_loss / max(total_batches, 1)
    return metrics


def train(args):
    set_seed(args.seed)
    splits, label_names = load_data(args)
    label_to_id = {label: i for i, label in enumerate(label_names)}
    id_to_label = {i: label for label, i in label_to_id.items()}
    vocab = build_vocab(splits["train"], args.lower, args.min_freq)

    train_data = NERDataset(splits["train"], vocab, label_to_id, args.lower, args.max_train_examples)
    valid_data = NERDataset(splits["validation"], vocab, label_to_id, args.lower, args.max_eval_examples)
    test_data = NERDataset(splits["test"], vocab, label_to_id, args.lower, args.max_eval_examples)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = BiLSTMTagger(
        vocab_size=len(vocab),
        num_labels=len(label_names),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pad_id=vocab[PAD_TOKEN],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_LABEL_ID)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
        for batch in progress:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            lengths = batch["lengths"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, lengths)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        valid_metrics = evaluate(model, valid_loader, id_to_label, device)
        train_loss = total_loss / max(len(train_loader), 1)
        row = {"epoch": epoch, "train_loss": train_loss, **{f"valid_{k}": v for k, v in valid_metrics.items()}}
        history.append(row)
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"valid_p={valid_metrics['precision']:.4f} "
            f"valid_r={valid_metrics['recall']:.4f} "
            f"valid_f1={valid_metrics['f1']:.4f}"
        )

        if valid_metrics["f1"] > best_f1:
            best_f1 = valid_metrics["f1"]
            torch.save(model.state_dict(), output_dir / "best_model.pt")

    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device))
    test_metrics = evaluate(model, test_loader, id_to_label, device)
    print(
        f"test precision={test_metrics['precision']:.4f} "
        f"recall={test_metrics['recall']:.4f} "
        f"f1={test_metrics['f1']:.4f} "
        f"token_acc={test_metrics['token_accuracy']:.4f}"
    )

    artifacts = {
        "args": vars(args),
        "vocab_size": len(vocab),
        "labels": label_names,
        "history": history,
        "test_metrics": test_metrics,
    }
    (output_dir / "metrics.json").write_text(json.dumps(artifacts, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "vocab.json").write_text(json.dumps(vocab, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "label_to_id.json").write_text(json.dumps(label_to_id, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="BiLSTM named entity recognition on CoNLL-2003.")
    parser.add_argument("--data_dir", default="", help="Directory with train/dev/test CoNLL files.")
    parser.add_argument("--use_sample", action="store_true", help="Run on a tiny built-in sample for smoke testing.")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--min_freq", type=int, default=1)
    parser.add_argument("--lower", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="")
    parser.add_argument("--max_train_examples", type=int, default=0)
    parser.add_argument("--max_eval_examples", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
