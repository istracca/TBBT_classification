import torch
import torch.nn as nn
from torch.utils.data import Dataset

class EmbeddingDataset(Dataset):
    """Dataset for pre-computed embeddings."""
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return {
            "embedding": self.X[idx],
            "label": self.y[idx]
        }

class TextDataset(Dataset):
    """Dataset for raw text inputs, which are tokenized on initialization."""
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx]),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx]),
            "labels": torch.tensor(self.labels[idx])
        }


class EmbeddingClassifier(nn.Module):
    """
    A simple classifier for pre-computed embeddings. Used for 'class' model_type.
    """
    def __init__(self, input_dim=384, num_classes=7, dropout_rate=0):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.model(x)

class SBERTWithClassifier(nn.Module):
    def __init__(self, sbert_model, num_classes, dropout_rate):
        super().__init__()
        self.sbert = sbert_model
        self.classifier = nn.Sequential(
            nn.Linear(384, 128),  # 384 è l'output embedding di MiniLM
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, input_ids, attention_mask):
        with torch.set_grad_enabled(self.training):
            outputs = self.sbert(input_ids=input_ids, attention_mask=attention_mask)
            pooled = outputs.last_hidden_state[:, 0]  # [CLS] token
            return self.classifier(pooled)