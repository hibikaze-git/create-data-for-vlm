from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

TRANSLATE_PROMPT = """You are a translator proficient in English and Japanese. Your task is to translate the following English text into Japanese, focusing on a natural and fluent result that avoids “translationese”. Please consider these points:
1. Keep proper nouns, brands, and geographical names in English.
2. Retain technical terms or jargon in English.
3. Use Japanese idiomatic expressions for English idioms or proverbs to ensure cultural relevance.
4. Ensure quotes or direct speech sound natural in Japanese, maintaining the original’s tone.
5. Answer only the translation result. No explanation of the translation results is required.

English text: {text}"""


class Phi3Manager:
    def __init__(self) -> None:
        model_id = "microsoft/Phi-3-medium-4k-instruct"
        # model_id = "microsoft/Phi-3-mini-4k-instruct"

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

    def translate(self, texts):
        user_messages_list = []
        assistant_messages_list = []

        for text in texts:
            messages = [
                {"role": "user", "content": TRANSLATE_PROMPT.format(text=text["user"])},
            ]

            user_messages_list.append(messages)

        for text in texts:
            messages = [
                {
                    "role": "user",
                    "content": TRANSLATE_PROMPT.format(text=text["assistant"]),
                },
            ]

            assistant_messages_list.append(messages)

        outputs_user = self.pipe(user_messages_list, **self.generation_args)
        outputs_assistant = self.pipe(assistant_messages_list, **self.generation_args)

        synthesis_dict = {
            "texts_ja": [
                {
                    "user": user[0]["generated_text"].strip(),
                    "assistant": assistant[0]["generated_text"].strip(),
                }
                for user, assistant in zip(outputs_user, outputs_assistant)
            ]
        }

        return synthesis_dict
