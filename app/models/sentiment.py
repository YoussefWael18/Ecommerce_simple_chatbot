"""
Sentiment and emotion analysis module.
Provides:
  - BiLSTMClassifier: PyTorch recurrent neural network for emotion detection.
  - SentimentAnalyzer: Unified inference wrapper supporting fine-tuned DistilRoBERTa
    (3-class sentiment: negative/neutral/positive) and BiLSTM (6-class emotion).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM classifier for emotion classification.
    Constructed using two stacked bidirectional LSTM layers followed by linear projection.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 64,
        num_classes: int = 6,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm1 = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_dim * 2, hidden_dim // 2, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        out, _ = self.lstm1(emb)
        out, (h_n, _) = self.lstm2(out)
        h_final = torch.cat((h_n[-2], h_n[-1]), dim=1)
        x = self.relu(self.fc1(h_final))
        x = self.dropout(x)
        return self.fc2(x)


def simple_tokenize(text: str) -> List[str]:
    """Lowercase whitespace tokenization for BiLSTM vocabulary lookup."""
    return text.lower().split()


def encode(text: str, vocab: Dict[str, int], max_len: int = 100) -> List[int]:
    """
    Encode text tokens to integer IDs with padding and out-of-vocabulary handling.

    Args:
        text: Input string.
        vocab: Dictionary mapping token strings to integer IDs.
        max_len: Fixed sequence length.

    Returns:
        List of integer token IDs padded to max_len.
    """
    OOV_IDX = 1
    PAD_IDX = 0
    ids = [vocab.get(tok, OOV_IDX) for tok in simple_tokenize(text)][:max_len]
    ids = ids + [PAD_IDX] * (max_len - len(ids))
    return ids


class SentimentAnalyzer:
    """
    Sentiment and emotion analyzer.
    Loads a fine-tuned RoBERTa sequence classifier for 3-class sentiment
    and optionally a BiLSTM model for 6-class emotion classification.
    """

    def __init__(
        self,
        roberta_model_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        bilstm_model_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize SentimentAnalyzer with model checkpoints and tokenizer.

        Args:
            roberta_model_path: Path to roberta_sentiment.pkl checkpoint.
            tokenizer_path: Path to roberta_tokenizer directory.
            bilstm_model_path: Optional path to bilstm_sentiment.pkl checkpoint.
            device: 'cuda', 'cpu', or None for auto-detection.
        """
        # Determine compute device
        if device is None:
            self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device_str = device
        self.device = torch.device(self.device_str)

        # Fallback to default paths from config if not explicitly provided
        if roberta_model_path is None or tokenizer_path is None:
            try:
                from app.config import ROBERTA_SENTIMENT_PATH, ROBERTA_TOKENIZER_PATH, BILSTM_SENTIMENT_PATH
                if roberta_model_path is None:
                    roberta_model_path = str(ROBERTA_SENTIMENT_PATH)
                if tokenizer_path is None:
                    tokenizer_path = str(ROBERTA_TOKENIZER_PATH)
                if bilstm_model_path is None and BILSTM_SENTIMENT_PATH.exists():
                    bilstm_model_path = str(BILSTM_SENTIMENT_PATH)
            except Exception:
                pass

        # ── 1. Load RoBERTa Model & Tokenizer ──────────────────────────────────
        if roberta_model_path is None or not Path(roberta_model_path).exists():
            raise FileNotFoundError(f"RoBERTa checkpoint not found at: {roberta_model_path}")
        if tokenizer_path is None or not Path(tokenizer_path).exists():
            raise FileNotFoundError(f"Tokenizer directory not found at: {tokenizer_path}")

        ckpt = torch.load(roberta_model_path, map_location=self.device, weights_only=False)

        # Reconstruct RoBERTa model directly from saved config (instant, offline)
        config = ckpt["config"]
        model = AutoModelForSequenceClassification.from_config(config)

        model.load_state_dict(ckpt["state_dict"])
        model.to(self.device).eval()
        self.roberta_model = model

        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # Store id2label mapping (default: 0=negative, 1=neutral, 2=positive)
        raw_id2label = ckpt.get("id2label", {0: "negative", 1: "neutral", 2: "positive"})
        self.id2label: Dict[int, str] = {int(k): v for k, v in raw_id2label.items()}

        # ── 2. Optionally Load BiLSTM Emotion Model ────────────────────────────
        self.bilstm_model: Optional[BiLSTMClassifier] = None
        self.bilstm_vocab: Dict[str, int] = {}
        self.bilstm_max_len: int = 100
        self.bilstm_label_names: List[str] = []

        if bilstm_model_path and Path(bilstm_model_path).exists():
            bilstm_ckpt = torch.load(bilstm_model_path, map_location=self.device, weights_only=False)
            self.bilstm_vocab = bilstm_ckpt.get("vocab", {})
            self.bilstm_max_len = bilstm_ckpt.get("max_len", 100)
            self.bilstm_label_names = bilstm_ckpt.get("label_names", [])
            model_config = bilstm_ckpt.get(
                "model_config",
                {"embed_dim": 64, "hidden_dim": 64, "num_classes": 6},
            )

            num_classes = model_config.get(
                "num_classes",
                len(self.bilstm_label_names) if self.bilstm_label_names else 6,
            )
            self.bilstm_model = BiLSTMClassifier(
                vocab_size=len(self.bilstm_vocab),
                embed_dim=model_config.get("embed_dim", 64),
                hidden_dim=model_config.get("hidden_dim", 64),
                num_classes=num_classes,
            )
            self.bilstm_model.load_state_dict(bilstm_ckpt["state_dict"])
            self.bilstm_model.to(self.device).eval()

    def predict(self, text: str, model_type: str = "roberta") -> Dict[str, Any]:
        """
        Run inference on the customer message using either RoBERTa or BiLSTM.

        Args:
            text: Customer message string.
            model_type: 'roberta' for 3-class sentiment (default),
                        or 'bilstm' for 6-class emotion.

        Returns:
            dict: {'sentiment': label, 'confidence': float} for roberta, or
                  {'emotion': label, 'confidence': float} for bilstm.
        """
        model_type = model_type.lower().strip()

        if model_type == "roberta":
            if self.roberta_model is None or self.tokenizer is None:
                raise RuntimeError("RoBERTa model or tokenizer is not loaded.")

            input_text = text if (text and text.strip()) else " "
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.roberta_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)[0]
                pred_idx = int(torch.argmax(probs).item())
                confidence = float(probs[pred_idx].item())

            label = self.id2label.get(pred_idx, str(pred_idx))
            return {"sentiment": label, "confidence": round(confidence, 4)}

        elif model_type == "bilstm":
            if self.bilstm_model is None:
                raise ValueError(
                    "BiLSTM model is not loaded. Please provide bilstm_model_path during initialization."
                )

            input_text = text if (text and text.strip()) else " "
            encoded = encode(input_text, self.bilstm_vocab, self.bilstm_max_len)
            input_tensor = torch.tensor([encoded], dtype=torch.long, device=self.device)

            with torch.no_grad():
                logits = self.bilstm_model(input_tensor)
                probs = torch.softmax(logits, dim=-1)[0]
                pred_idx = int(torch.argmax(probs).item())
                confidence = float(probs[pred_idx].item())

            if self.bilstm_label_names and pred_idx < len(self.bilstm_label_names):
                label = self.bilstm_label_names[pred_idx]
            else:
                label = str(pred_idx)

            return {"emotion": label, "confidence": round(confidence, 4)}

        else:
            raise ValueError(
                f"Unsupported model_type: '{model_type}'. Expected 'roberta' or 'bilstm'."
            )

