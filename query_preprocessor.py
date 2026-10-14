import os
import sys

# Dynamic imports (supports both root-level and data_operations/ folder setups)
try:
    from regional_expander import RegionalSynonymExpander
except ImportError:
    from data_operations.regional_expander import RegionalSynonymExpander

try:
    from fuzzy_corrector import DomainFuzzyCorrector
except ImportError:
    try:
        from data_operations.fuzzy_corrector import DomainFuzzyCorrector
    except ImportError:
        DomainFuzzyCorrector = None


class AgroQueryPreprocessor:
    def __init__(self, dataset_path: str = None):
        """
        Unified preprocessing pipeline combining Fuzzy Spell Correction
        and Regional Synonym Expansion.
        """
        self.expander = RegionalSynonymExpander()
        
        # Safe fallback if rapidfuzz or fuzzy_corrector is missing
        if DomainFuzzyCorrector:
            if dataset_path and os.path.exists(dataset_path):
                self.corrector = DomainFuzzyCorrector.build_from_dataset(dataset_path)
            else:
                self.corrector = DomainFuzzyCorrector()
        else:
            self.corrector = None

    def process(self, raw_query: str) -> str:
        """
        Executes end-to-end transformation:
        1. Corrects domain spelling errors (e.g., 'chlorpyrifs' -> 'chlorpyrifos')
        2. Expands vernacular terms (e.g., 'gehu' -> 'wheat')
        """
        if not raw_query or not raw_query.strip():
            return ""

        processed_text = raw_query.strip()

        # Step 1: Apply Spell Correction (if module is loaded)
        if self.corrector:
            try:
                processed_text = self.corrector.correct_query(processed_text)
            except Exception:
                pass  # Fall back gracefully on failure

        # Step 2: Apply Vernacular Synonym Expansion
        processed_text = self.expander.expand(processed_text)

        return processed_text


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    DATASET = os.path.join(current_dir, "datasetsQueries", "Question-Answer", "questionsv4kaggle.csv")
    
    pipeline = AgroQueryPreprocessor(dataset_path=DATASET)
    
    test_query = "gehu me keeda lag gaya chlorpyrifs dose"
    print(f"Raw Input : {test_query}")
    print(f"Processed : {pipeline.process(test_query)}")