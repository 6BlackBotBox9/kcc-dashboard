"""
regional_expander.py — Translates Hindi (Devanagari) & Hinglish (Roman) to English
"""
import re

class RegionalSynonymExpander:
    def __init__(self, custom_dict: dict = None):
        # Comprehensive bilingual mapping: Devanagari Hindi + Roman Hinglish -> English
        self.devanagari_map = {
            'गेहूं': 'wheat', 'गेहू': 'wheat', 'गेहुं': 'wheat', 'गेहु': 'wheat',
            'धान': 'paddy rice', 'चावल': 'paddy rice',
            'आलू': 'potato', 'आलु': 'potato',
            'सरसों': 'mustard', 'सरसो': 'mustard', 'राई': 'mustard',
            'कपास': 'cotton', 'रूई': 'cotton',
            'टमाटर': 'tomato',
            'प्याज': 'onion', 'प्याज़': 'onion',
            'मक्का': 'maize corn', 'मकई': 'maize corn',
            'गन्ना': 'sugarcane', 'ईख': 'sugarcane',
            'चना': 'gram chickpea', 'चने': 'gram chickpea',
            'बैंगन': 'brinjal eggplant', 'भिंडी': 'okra ladyfinger',
            'आम': 'mango', 'केला': 'banana',
            'सोयाबीन': 'soybean', 'मूंगफली': 'groundnut peanut',
            'मिर्च': 'chilli pepper', 'मिर्ची': 'chilli pepper',

            'कीड़ा': 'pest insect', 'कीड़े': 'pest insect', 'कीट': 'pest insect', 'इल्ली': 'caterpillar pest',
            'बीमारी': 'disease infection', 'रोग': 'disease', 'फफूंद': 'fungus blight', 'उकठा': 'wilt rot',
            'पत्ता': 'leaf', 'पत्ती': 'leaf', 'पत्ते': 'leaves', 'पत्तियां': 'leaves', 'पत्तिया': 'leaves',
            'फल': 'fruit', 'फूल': 'flower', 'जड़': 'root', 'तना': 'stem',
            'पीलापन': 'yellowing', 'पीला': 'yellow', 'पीली': 'yellow', 'काला': 'black', 'सफेद': 'white',
            'धब्बा': 'spot', 'धब्बे': 'spots', 'सुंडी': 'borer worm',
            'दवा': 'pesticide medicine', 'दवाई': 'pesticide medicine', 'छिड़काव': 'spray application',
            'खाद': 'fertilizer manure', 'उर्वरक': 'fertilizer', 'यूरिया': 'urea',
            'पानी': 'water irrigation', 'सिंचाई': 'irrigation',
            'नुकसान': 'damage loss', 'उपचार': 'treatment cure', 'समाधान': 'solution control',
            'भाव': 'mandi market price rate', 'दाम': 'price rate', 'रेट': 'price rate',
            'सब्सिडी': 'subsidy scheme', 'योजना': 'scheme policy', 'बीमा': 'insurance claim pmfby'
        }

        self.roman_map = {
            'gehu': 'wheat', 'gehun': 'wheat', 'gihu': 'wheat',
            'dhan': 'paddy rice', 'chawal': 'paddy rice', 'chaawal': 'paddy rice',
            'aalu': 'potato', 'aloo': 'potato',
            'sarsom': 'mustard', 'sarson': 'mustard', 'sarso': 'mustard',
            'kapas': 'cotton', 'kapaas': 'cotton',
            'tamatar': 'tomato', 'tamatr': 'tomato',
            'pyaz': 'onion', 'pyaaj': 'onion', 'piaz': 'onion',
            'makka': 'maize corn', 'makkai': 'maize corn',
            'ganna': 'sugarcane', 'gannaa': 'sugarcane',
            'chana': 'gram chickpea', 'channa': 'gram chickpea',
            'baingan': 'brinjal eggplant', 'bhindi': 'okra ladyfinger',
            'aam': 'mango', 'kela': 'banana',
            'soyabean': 'soybean', 'mungfali': 'groundnut peanut', 'moongfali': 'groundnut peanut',
            'mirch': 'chilli pepper', 'mirchi': 'chilli pepper',

            'kida': 'pest insect', 'keeda': 'pest insect', 'keede': 'pest insect', 'kit': 'pest insect',
            'bimari': 'disease infection', 'beemari': 'disease infection', 'rog': 'disease',
            'patta': 'leaf', 'patti': 'leaf', 'pattiya': 'leaves', 'pattiyan': 'leaves', 'patte': 'leaves',
            'phal': 'fruit', 'phool': 'flower', 'jad': 'root', 'tana': 'stem',
            'pilapan': 'yellowing', 'peelapan': 'yellowing', 'pila': 'yellow', 'pili': 'yellow',
            'dhabba': 'spot', 'dhabbe': 'spots', 'sundi': 'borer worm',
            'dawa': 'pesticide medicine', 'dawai': 'pesticide medicine', 'dawaai': 'pesticide medicine', 'chhidkaw': 'spray',
            'khad': 'fertilizer manure', 'khat': 'fertilizer manure', 'paani': 'water irrigation', 'pani': 'water irrigation',
            'bhav': 'mandi market price rate', 'daam': 'price rate', 'rate': 'price rate',
            'subsidy': 'subsidy scheme', 'yojana': 'scheme policy', 'bima': 'insurance claim pmfby'
        }

        if custom_dict:
            self.roman_map.update(custom_dict)

        # Devanagari regex without  (Devanagari characters are not matched by ASCII )
        self.devanagari_patterns = [
            (re.compile(re.escape(k), re.IGNORECASE), v)
            for k, v in sorted(self.devanagari_map.items(), key=lambda x: len(x[0]), reverse=True)
        ]

        # Roman Hinglish regex with 
        self.roman_patterns = [
            (re.compile(r'\b' + re.escape(k) + r'\b', re.IGNORECASE), v)
            for k, v in sorted(self.roman_map.items(), key=lambda x: len(x[0]), reverse=True)
        ]

    def expand(self, query: str) -> str:
        """Translates regional input terms into standard BM25 index terms."""
        if not query:
            return ''
        text = query.lower()
        # 1. Apply Devanagari translation first
        for pattern, replacement in self.devanagari_patterns:
            text = pattern.sub(f' {replacement} ', text)
        # 2. Apply Roman Hinglish translation
        for pattern, replacement in self.roman_patterns:
            text = pattern.sub(replacement, text)
        # Clean multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text
