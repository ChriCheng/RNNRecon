import argparse
import json
import random
import sys
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
PAD_CHAR = "<PAD_CHAR>"
UNK_CHAR = "<UNK_CHAR>"


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


def load_from_huggingface(dataset_name: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets with `python3 -m pip install datasets`, or pass --data_dir/--use_sample.") from exc

    dataset = load_dataset(dataset_name, trust_remote_code=True)
    label_names = dataset["train"].features["ner_tags"].feature.names
    splits = {
        split: [
            (row["tokens"], [label_names[label_id] for label_id in row["ner_tags"]])
            for row in dataset[split]
        ]
        for split in ["train", "validation", "test"]
    }
    return splits, list(label_names)


def load_data(args):
    if args.use_sample:
        return sample_data()
    if args.data_dir:
        return load_from_conll_dir(Path(args.data_dir))
    return load_from_huggingface(args.dataset_name)


def build_vocab(examples, lower: bool, min_freq: int):
    counter = Counter()
    for tokens, _ in examples:
        counter.update(normalize_token(token, lower) for token in tokens)

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, freq in counter.most_common():
        if freq >= min_freq and token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def find_singleton_token_ids(examples, vocab, lower: bool):
    counter = Counter()
    for tokens, _ in examples:
        counter.update(normalize_token(token, lower) for token in tokens)
    return {vocab[token] for token, freq in counter.items() if freq == 1 and token in vocab}


def build_char_vocab(examples):
    chars = {PAD_CHAR: 0, UNK_CHAR: 1}
    for tokens, _ in examples:
        for token in tokens:
            for char in token:
                if char not in chars:
                    chars[char] = len(chars)
    return chars


def load_pretrained_embeddings(path: str, vocab):
    embedding_path = Path(path)
    if not embedding_path.exists():
        raise FileNotFoundError(f"Pretrained embedding file not found: {embedding_path}")

    lookup = set(vocab)
    lookup.update(token.lower() for token in vocab)
    vectors = {}
    embedding_dim = None

    with embedding_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line_number, line in enumerate(f, start=1):
            parts = line.rstrip().split()
            if not parts:
                continue
            if line_number == 1 and len(parts) == 2 and all(part.isdigit() for part in parts):
                continue
            token, values = parts[0], parts[1:]
            if token not in lookup:
                continue
            if embedding_dim is None:
                embedding_dim = len(values)
            if len(values) != embedding_dim:
                continue
            try:
                vectors[token] = torch.tensor([float(value) for value in values], dtype=torch.float)
            except ValueError:
                continue

    if embedding_dim is None:
        raise ValueError(f"No vocabulary tokens matched pretrained embedding file: {embedding_path}")

    weights = torch.empty(len(vocab), embedding_dim)
    nn.init.uniform_(weights, -0.05, 0.05)
    weights[vocab[PAD_TOKEN]].zero_()
    matched = 0
    for token, idx in vocab.items():
        vector = vectors.get(token)
        if vector is None:
            vector = vectors.get(token.lower())
        if vector is not None:
            weights[idx] = vector
            matched += 1
    print(f"loaded_pretrained_embeddings={matched}/{len(vocab)} dim={embedding_dim}", flush=True)
    return weights


class NERDataset(Dataset):
    def __init__(self, examples, vocab, label_to_id, lower: bool, char_vocab=None, max_word_len: int = 24, max_examples: int = 0):
        if max_examples:
            examples = examples[:max_examples]
        self.rows = []
        for tokens, labels in examples:
            token_ids = [vocab.get(normalize_token(token, lower), vocab[UNK_TOKEN]) for token in tokens]
            label_ids = [label_to_id[label] for label in labels]
            char_ids = None
            if char_vocab is not None:
                char_ids = []
                for token in tokens:
                    ids = [char_vocab.get(char, char_vocab[UNK_CHAR]) for char in token[:max_word_len]]
                    char_ids.append(ids or [char_vocab[UNK_CHAR]])
            self.rows.append((tokens, token_ids, label_ids, char_ids))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def collate_batch(batch):
    tokens, token_ids, label_ids, char_ids = zip(*batch)
    lengths = torch.tensor([len(ids) for ids in token_ids], dtype=torch.long)
    max_len = max(lengths).item()

    input_ids = torch.zeros((len(batch), max_len), dtype=torch.long)
    labels = torch.full((len(batch), max_len), PAD_LABEL_ID, dtype=torch.long)
    mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    batch_char_ids = None
    if char_ids[0] is not None:
        max_word_len = max(len(chars) for row in char_ids for chars in row)
        batch_char_ids = torch.zeros((len(batch), max_len, max_word_len), dtype=torch.long)

    for i, (ids, labs, chars) in enumerate(zip(token_ids, label_ids, char_ids)):
        size = len(ids)
        input_ids[i, :size] = torch.tensor(ids, dtype=torch.long)
        labels[i, :size] = torch.tensor(labs, dtype=torch.long)
        mask[i, :size] = True
        if batch_char_ids is not None:
            for j, word_chars in enumerate(chars):
                batch_char_ids[i, j, :len(word_chars)] = torch.tensor(word_chars, dtype=torch.long)
    return {
        "tokens": tokens,
        "input_ids": input_ids,
        "char_ids": batch_char_ids,
        "labels": labels,
        "lengths": lengths,
        "mask": mask,
    }


class CharCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, out_channels, kernel_size, dropout, pad_id):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.conv = nn.Conv1d(embedding_dim, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, char_ids):
        batch_size, seq_len, word_len = char_ids.shape
        flat = char_ids.view(batch_size * seq_len, word_len)
        embedded = self.embedding(flat).transpose(1, 2)
        encoded = torch.relu(self.conv(embedded))
        pooled = torch.max(encoded, dim=-1).values
        return self.dropout(pooled).view(batch_size, seq_len, -1)


class LinearChainCRF(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        self.num_tags = num_tags
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def forward(self, emissions, tags, mask):
        tags = tags.masked_fill(~mask, 0)
        return self._score_sentence(emissions, tags, mask) - self._compute_normalizer(emissions, mask)

    def _score_sentence(self, emissions, tags, mask):
        batch_size, seq_len, _ = emissions.shape
        score = self.start_transitions[tags[:, 0]]
        score += emissions[torch.arange(batch_size, device=emissions.device), 0, tags[:, 0]]

        for t in range(1, seq_len):
            previous_tag = tags[:, t - 1]
            current_tag = tags[:, t]
            step_score = self.transitions[previous_tag, current_tag]
            step_score += emissions[torch.arange(batch_size, device=emissions.device), t, current_tag]
            score += step_score * mask[:, t]

        lengths = mask.long().sum(dim=1) - 1
        last_tags = tags[torch.arange(batch_size, device=emissions.device), lengths]
        score += self.end_transitions[last_tags]
        return score

    def _compute_normalizer(self, emissions, mask):
        score = self.start_transitions + emissions[:, 0]
        for t in range(1, emissions.size(1)):
            broadcast_score = score.unsqueeze(2)
            broadcast_emission = emissions[:, t].unsqueeze(1)
            next_score = torch.logsumexp(broadcast_score + self.transitions + broadcast_emission, dim=1)
            score = torch.where(mask[:, t].unsqueeze(1), next_score, score)
        score += self.end_transitions
        return torch.logsumexp(score, dim=1)

    def decode(self, emissions, mask):
        score = self.start_transitions + emissions[:, 0]
        history = []

        for t in range(1, emissions.size(1)):
            next_score = score.unsqueeze(2) + self.transitions + emissions[:, t].unsqueeze(1)
            best_score, best_path = next_score.max(dim=1)
            score = torch.where(mask[:, t].unsqueeze(1), best_score, score)
            history.append(best_path)

        score += self.end_transitions
        best_last_tags = score.argmax(dim=1)
        lengths = mask.long().sum(dim=1)
        best_paths = []
        for i in range(emissions.size(0)):
            length = lengths[i].item()
            best_tag = best_last_tags[i]
            path = [best_tag.item()]
            for hist in reversed(history[: length - 1]):
                best_tag = hist[i][best_tag]
                path.append(best_tag.item())
            best_paths.append(list(reversed(path)))
        return best_paths


class BiLSTMTagger(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_labels,
        embedding_dim,
        hidden_dim,
        num_layers,
        dropout,
        pad_id,
        char_vocab_size=0,
        char_embedding_dim=30,
        char_out_channels=30,
        char_kernel_size=3,
        char_pad_id=0,
        use_crf=True,
        pretrained_embeddings=None,
        freeze_embeddings=False,
    ):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = not freeze_embeddings
        self.char_encoder = None
        lstm_input_dim = embedding_dim
        if char_vocab_size:
            self.char_encoder = CharCNN(
                vocab_size=char_vocab_size,
                embedding_dim=char_embedding_dim,
                out_channels=char_out_channels,
                kernel_size=char_kernel_size,
                dropout=dropout,
                pad_id=char_pad_id,
            )
            lstm_input_dim += char_out_channels
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout,
        )
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)
        self.crf = LinearChainCRF(num_labels) if use_crf else None

    def forward(self, input_ids, lengths, char_ids=None):
        embeddings = self.embedding(input_ids)
        if self.char_encoder is not None:
            if char_ids is None:
                raise ValueError("char_ids must be provided when character encoder is enabled.")
            embeddings = torch.cat([embeddings, self.char_encoder(char_ids)], dim=-1)
        embeddings = self.dropout(embeddings)
        packed = nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        return self.classifier(self.dropout(output))

    def loss(self, emissions, labels, mask):
        if self.crf is not None:
            return -self.crf(emissions, labels, mask).mean()
        return nn.functional.cross_entropy(
            emissions.reshape(-1, emissions.size(-1)),
            labels.reshape(-1),
            ignore_index=PAD_LABEL_ID,
        )

    def decode(self, emissions, mask):
        if self.crf is not None:
            return self.crf.decode(emissions, mask)
        return emissions.argmax(dim=-1).cpu().tolist()


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

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        char_ids = batch["char_ids"].to(device) if batch["char_ids"] is not None else None
        labels = batch["labels"].to(device)
        lengths = batch["lengths"].to(device)
        mask = batch["mask"].to(device)
        emissions = model(input_ids, lengths, char_ids)
        loss = model.loss(emissions, labels, mask)
        total_loss += loss.item()
        total_batches += 1

        predictions = model.decode(emissions, mask)
        for pred_row, gold_row in zip(predictions, batch["labels"]):
            gold = []
            pred = []
            for pred_id, gold_id in zip(list(pred_row), gold_row.tolist()):
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
    singleton_token_ids = find_singleton_token_ids(splits["train"], vocab, args.lower)
    char_vocab = build_char_vocab(splits["train"]) if not args.no_char else None
    pretrained_embeddings = None
    embedding_dim = args.embedding_dim
    if args.pretrained_embeddings:
        pretrained_embeddings = load_pretrained_embeddings(args.pretrained_embeddings, vocab)
        embedding_dim = pretrained_embeddings.size(1)

    train_data = NERDataset(
        splits["train"], vocab, label_to_id, args.lower, char_vocab, args.max_word_len, args.max_train_examples
    )
    valid_data = NERDataset(
        splits["validation"], vocab, label_to_id, args.lower, char_vocab, args.max_word_len, args.max_eval_examples
    )
    test_data = NERDataset(
        splits["test"], vocab, label_to_id, args.lower, char_vocab, args.max_word_len, args.max_eval_examples
    )

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = BiLSTMTagger(
        vocab_size=len(vocab),
        num_labels=len(label_names),
        embedding_dim=embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pad_id=vocab[PAD_TOKEN],
        char_vocab_size=len(char_vocab) if char_vocab is not None else 0,
        char_embedding_dim=args.char_embedding_dim,
        char_out_channels=args.char_out_channels,
        char_kernel_size=args.char_kernel_size,
        char_pad_id=char_vocab[PAD_CHAR] if char_vocab is not None else 0,
        use_crf=not args.no_crf,
        pretrained_embeddings=pretrained_embeddings,
        freeze_embeddings=args.freeze_embeddings,
    ).to(device)

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = None
    if args.lr_decay_patience > 0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=args.lr_decay_factor,
            patience=args.lr_decay_patience,
            min_lr=args.min_lr,
        )

    singleton_lookup = torch.zeros(len(vocab), dtype=torch.bool, device=device)
    if singleton_token_ids:
        singleton_lookup[list(singleton_token_ids)] = True

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        progress = tqdm(
            train_loader,
            desc=f"epoch {epoch}/{args.epochs}",
            leave=False,
            disable=args.no_progress or not sys.stderr.isatty(),
        )
        for batch in progress:
            input_ids = batch["input_ids"].to(device)
            char_ids = batch["char_ids"].to(device) if batch["char_ids"] is not None else None
            labels = batch["labels"].to(device)
            lengths = batch["lengths"].to(device)
            mask = batch["mask"].to(device)
            if args.word_dropout > 0:
                candidates = mask
                if args.singleton_word_dropout:
                    candidates = candidates & singleton_lookup[input_ids]
                drop_mask = candidates & (torch.rand(input_ids.shape, device=device) < args.word_dropout)
                if drop_mask.any():
                    input_ids = input_ids.clone()
                    input_ids[drop_mask] = vocab[UNK_TOKEN]

            optimizer.zero_grad()
            emissions = model(input_ids, lengths, char_ids)
            loss = model.loss(emissions, labels, mask)
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
            f"valid_f1={valid_metrics['f1']:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.6g}",
            flush=True,
        )

        improved = valid_metrics["f1"] > best_f1 + args.min_delta
        if improved:
            best_f1 = valid_metrics["f1"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), output_dir / "best_model.pt")
        else:
            epochs_without_improvement += 1

        if scheduler is not None:
            scheduler.step(valid_metrics["f1"])

        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            print(f"early_stop epoch={epoch} best_epoch={best_epoch} best_valid_f1={best_f1:.4f}", flush=True)
            break

    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device))
    test_metrics = evaluate(model, test_loader, id_to_label, device)
    print(
        f"test precision={test_metrics['precision']:.4f} "
        f"recall={test_metrics['recall']:.4f} "
        f"f1={test_metrics['f1']:.4f} "
        f"token_acc={test_metrics['token_accuracy']:.4f}",
        flush=True,
    )

    artifacts = {
        "args": vars(args),
        "vocab_size": len(vocab),
        "labels": label_names,
        "history": history,
        "best_epoch": best_epoch,
        "best_valid_f1": best_f1,
        "test_metrics": test_metrics,
    }
    (output_dir / "metrics.json").write_text(json.dumps(artifacts, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "vocab.json").write_text(json.dumps(vocab, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "label_to_id.json").write_text(json.dumps(label_to_id, indent=2, ensure_ascii=False), encoding="utf-8")
    if char_vocab is not None:
        (output_dir / "char_vocab.json").write_text(json.dumps(char_vocab, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="BiLSTM named entity recognition on CoNLL-2003.")
    parser.add_argument("--dataset_name", default="conll2003", help="Hugging Face datasets name.")
    parser.add_argument("--data_dir", default="", help="Directory with train/dev/test CoNLL files.")
    parser.add_argument("--use_sample", action="store_true", help="Run on a tiny built-in sample for smoke testing.")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--pretrained_embeddings", default="", help="Path to GloVe/SENNA-style text embeddings.")
    parser.add_argument("--freeze_embeddings", action="store_true", help="Keep pretrained word embeddings fixed.")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--char_embedding_dim", type=int, default=30)
    parser.add_argument("--char_out_channels", type=int, default=30)
    parser.add_argument("--char_kernel_size", type=int, default=3)
    parser.add_argument("--max_word_len", type=int, default=24)
    parser.add_argument("--no_char", action="store_true", help="Disable character CNN features.")
    parser.add_argument("--no_crf", action="store_true", help="Disable CRF decoding and use token-level softmax.")
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--nesterov", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--word_dropout", type=float, default=0.0)
    parser.add_argument(
        "--singleton_word_dropout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply word dropout only to singleton train tokens by default.",
    )
    parser.add_argument("--lr_decay_patience", type=int, default=0)
    parser.add_argument("--lr_decay_factor", type=float, default=0.5)
    parser.add_argument("--min_lr", type=float, default=1e-5)
    parser.add_argument("--early_stop_patience", type=int, default=0)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--min_freq", type=int, default=1)
    parser.add_argument("--lower", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="")
    parser.add_argument("--no_progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument("--max_train_examples", type=int, default=0)
    parser.add_argument("--max_eval_examples", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
