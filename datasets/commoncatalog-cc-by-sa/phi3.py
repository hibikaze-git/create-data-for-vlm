from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

TRANSLATE_PROMPT = """You are a translator proficient in English and Japanese. Your task is to translate the following English text into Japanese, focusing on a natural and fluent result that avoids “translationese.” Please consider these points:
1. Keep proper nouns, brands, and geographical names in English.
2. Retain technical terms or jargon in English.
3. Use Japanese idiomatic expressions for English idioms or proverbs to ensure cultural relevance.
4. Ensure quotes or direct speech sound natural in Japanese, maintaining the original’s tone.
5. Answer only the translation result. No explanation of the translation results is required.

English text: {text}\n"""


class Phi3Manager:
    def __init__(self) -> None:
        model_id = "microsoft/Phi-3-medium-4k-instruct"
        #model_id = "microsoft/Phi-3-mini-4k-instruct"

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cuda",
            trust_remote_code=True,
            torch_dtype="auto",
            cache_dir="../../cache",
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )

        self.generation_args = {
            "max_new_tokens": 4096,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }

    def translate(self, text_dict):
        messages_list = []

        for key, text in text_dict.items():
            messages = [
                {"role": "user", "content": TRANSLATE_PROMPT.format(text=text)},
            ]

            messages_list.append(messages)

        outputs = self.pipe(messages_list, **self.generation_args)

        synthesis_dict = {
            f"{key}_ja": output[0]["generated_text"].strip()
            for key, output in zip(text_dict.keys(), outputs)
        }

        return synthesis_dict
