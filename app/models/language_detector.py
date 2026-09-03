"""
Language detection module for customer support messages.
Loads a scikit-learn Pipeline (TfidfVectorizer + MultinomialNB) trained on multilingual data.
"""

import pickle
import warnings
from pathlib import Path
from typing import Optional, Tuple


class LanguageDetector:
    """
    Language detector using a pre-trained scikit-learn Pipeline
    (TfidfVectorizer + MultinomialNB) to classify customer queries by language code.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the LanguageDetector by loading the pickled pipeline.

        Args:
            model_path: Path to the pickled pipeline file. Defaults to
                        the path configured in app.config or 'Model_pickle/lang_classifier .pkl'.
        """
        if model_path is None:
            try:
                from app.config import LANG_CLASSIFIER_PATH
                model_path = str(LANG_CLASSIFIER_PATH)
            except Exception:
                model_path = str(
                    Path(__file__).resolve().parent.parent.parent
                    / "Model_pickle"
                    / "lang_classifier .pkl"
                )

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Language classifier pickle file not found at: {self.model_path}"
            )

        # Suppress scikit-learn version mismatch warnings on unpickling
        with warnings.catch_warnings():
            try:
                from sklearn.exceptions import InconsistentVersionWarning
                warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
            except ImportError:
                warnings.simplefilter("ignore")

            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)

    def predict(self, text: str) -> str:
        """
        Predict the language code for the given text.

        Args:
            text: Input string to detect language for.

        Returns:
            Detected language code string (e.g., 'en', 'de', 'es', 'fr', 'it').
        """
        if not text or not str(text).strip():
            return "en"

        cleaned_text = str(text).strip()
        preds = self.model.predict([cleaned_text])
        return str(preds[0])

    def predict_proba(self, text: str) -> Tuple[str, float]:
        """
        Predict the language code along with its confidence score.

        Args:
            text: Input string to detect language for.

        Returns:
            Tuple of (detected_language_code, confidence_score).
        """
        if not text or not str(text).strip():
            return "en", 1.0

        cleaned_text = str(text).strip()
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba([cleaned_text])[0]
            max_idx = probs.argmax()
            predicted_lang = str(self.model.classes_[max_idx])
            confidence = float(probs[max_idx])
            return predicted_lang, round(confidence, 4)

        return self.predict(cleaned_text), 1.0

