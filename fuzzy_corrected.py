import re
import pandas as pd
try:
    from rapidfuzz import process, fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False


class DomainFuzzyCorrector:
    def __init__(self, vocabulary: set = None):
        # Fallback agricultural domain vocabulary
        default_vocab = {
            "wheat", "paddy", "rice", "cotton", "mustard", "potato", "tomato", 
            "maize", "sugarcane", "gram", "onion", "chilli", "brinjal",
            "fertilizer", "pesticide", "insecticide", "fungicide", "herbicide",
            "urea", "dap", "chlorpyrifos", "glyphosate", "monocrotophos", "mancozeb",
            "yellowing", "blight", "wilt", "borer", "aphid", "whitefly", "rust"
        }
        self.vocab = list(vocabulary) if vocabulary else list(default_vocab)

    @classmethod
    def build_from_dataset(cls, csv_path: str, column_name: str = "questions"):
        """Extracts unique domain words (>3 characters) directly from dataset."""
        try:
            df = pd.read_csv(csv_path)
            words = set()
            for text in df[column_name].dropna().astype(str):
                extracted = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
                words.update(extracted)
            return cls(vocabulary=words)
        except Exception:
            return cls()

    def correct_token(self, token: str, score_cutoff: float = 82.0) -> str:
        """Corrects typos in individual tokens using C++ string matching."""
        if not HAS_RAPIDFUZZ or len(token) <= 3 or token.isdigit() or token in self.vocab:
            return token
            
        match = process.extractOne(
            token, 
            self.vocab, 
            scorer=fuzz.WRatio, 
            score_cutoff=score_cutoff
        )
        return match[0] if match else token

    def correct_query(self, query: str, score_cutoff: float = 82.0) -> str:
        """Splits input into tokens and corrects typos in place."""
        if not query:
            return ""
        tokens = query.lower().split()
        corrected_tokens = [self.correct_token(t, score_cutoff) for t in tokens]
        return " ".join(corrected_tokens)