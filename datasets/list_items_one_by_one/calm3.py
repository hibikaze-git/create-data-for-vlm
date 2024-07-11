from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

TRANSLATE_LABEL_PROMPT = """以下の英単語または英語のフレーズを日本語に翻訳してください。翻訳結果のみ回答してください。
{text}"""


TRANSLATE_PROMPT = """以下の英語のシーンの説明を日本語に翻訳してください。
{text}"""


class Calm3Manager:
    def __init__(self) -> None:
        model_id = "cyberagent/calm3-22b-chat"

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype="auto",
            cache_dir="../../cache",
            load_in_8bit=True
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
