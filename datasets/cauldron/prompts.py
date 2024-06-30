TRANSLATE_PROMPT = """You are a translator proficient in English and Japanese. Your task is to translate the following English text into Japanese, focusing on a natural and fluent result that avoids “translationese”. Please consider these points:
1. Keep proper nouns, brands, and geographical names in English.
2. Retain technical terms or jargon in English.
3. Use Japanese idiomatic expressions for English idioms or proverbs to ensure cultural relevance.
4. Ensure quotes or direct speech sound natural in Japanese, maintaining the original’s tone.
5. Answer only the translation result. No explanation of the translation results is required.

English text: {text}"""


TRANSLATE_PROMPT_AOKVQA = """You are a translator proficient in English and Japanese. Your task is to translate the following English text into Japanese, focusing on a natural and fluent result that avoids “translationese”. Please consider these points:
1. Keep proper nouns, brands, and geographical names in English.
2. Retain technical terms or jargon in English.
3. Answer only the translation result. No explanation of the translation results is required.
4. Try to use natural Japanese.

English text: {text}"""


TRANSLATE_PROMPT_CHARTQA = """You are a translator proficient in English and Japanese. Your task is to translate the following English text into Japanese, focusing on a natural and fluent result that avoids “translationese”. Please consider these points:
1. Keep proper nouns, brands, and geographical names in English.
2. Retain technical terms or jargon in English.
3. Answer only the translation result. No explanation of the translation results is required.

English text: {text}"""


PROMPT_DICT = {
    "aokvqa": TRANSLATE_PROMPT_AOKVQA,
    "vqarad": TRANSLATE_PROMPT,
    "vistext": TRANSLATE_PROMPT,
    "chartqa": TRANSLATE_PROMPT_CHARTQA,
    "plotqa": TRANSLATE_PROMPT,
    "plotqa_shuffle": TRANSLATE_PROMPT,
    "mapqa": TRANSLATE_PROMPT,
    "tat_qa": TRANSLATE_PROMPT,
    "hitab": TRANSLATE_PROMPT,
    "finqa": TRANSLATE_PROMPT,
    "robut_wikisql": TRANSLATE_PROMPT,
    "robut_sqa": TRANSLATE_PROMPT,
    "robut_wtq": TRANSLATE_PROMPT,
    "clevr": TRANSLATE_PROMPT,
    "raven": TRANSLATE_PROMPT,
    "intergps": TRANSLATE_PROMPT,
    "ai2d": TRANSLATE_PROMPT,
    "websight": TRANSLATE_PROMPT,
}
