from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

TRANSLATE_LABEL_PROMPT = """You are a translator proficient in English and Japanese. Your task is to translate the following English word or phrase into Japanese. Please consider these points:
1. Keep proper nouns, brands, and geographical names in English.
2. Answer only the translation result.
3. There is no problem using Japanese English.
4. Below is a sentence that describes the scene. We will translate words and phrases that fit the scene.
{detailed}

English word or phrase: {text}"""


TRANSLATE_PROMPT = """You are a translator proficient in English and Japanese. Your task is to translate the following English Scene description into Japanese, focusing on a natural and fluent result that avoids “translationese”. Please consider these points:
1. Keep proper nouns, brands, and geographical names in English.
2. Retain technical terms or jargon in English.
3. Answer only the translation result.
4. Try to use natural Japanese.

English Scene description: {text}"""


class Phi3Manager:
    def __init__(self) -> None:
        # model_id = "microsoft/Phi-3-medium-4k-instruct"
        model_id = "microsoft/Phi-3-mini-4k-instruct"

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
            batch_size=1,
        )

        self.generation_args = {
            "max_new_tokens": 4096,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }

    def translate(self, synthesis_dict):
        messages_list = []

        labels = synthesis_dict["labels"]

        # label
        for key, item in labels.items():
            messages = [
                {
                    "role": "user",
                    "content": TRANSLATE_LABEL_PROMPT.format(
                        text=item["label"],
                        detailed=synthesis_dict["detailed"].replace("\n", ""),
                    ),
                },
            ]

            messages_list.append(messages)

        print(messages_list[0])

        outputs = self.pipe(messages_list, **self.generation_args)

        labels_ja = {
            key: output[0]["generated_text"].strip()
            for key, output in zip(labels.keys(), outputs)
        }

        messages_list = []

        messages = [
            {
                "role": "user",
                "content": TRANSLATE_PROMPT.format(
                    text=synthesis_dict["detailed"]
                ),
            },
        ]

        messages_list.append(messages)

        outputs = self.pipe(messages_list, **self.generation_args)

        detailed_ja = outputs[0][0]["generated_text"].strip()

        synthesis_dict["labels_ja"] = labels_ja
        synthesis_dict["detailed_ja"] = detailed_ja

        return synthesis_dict
