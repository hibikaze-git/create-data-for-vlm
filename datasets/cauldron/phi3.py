from prompts import PROMPT_DICT
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

REMOVE_NEW_LINE = [
    "finqa",
]

EXTRACT_QUESTION = ["aokvqa"]


class Phi3Manager:
    def __init__(self, subset_name) -> None:
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
            batch_size=4 if subset_name == "plotqa_shuffle" else 1,
        )

        self.generation_args = {
            "max_new_tokens": 4096,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }

        self.prompt = PROMPT_DICT[subset_name]

        self.remove_new_line = True if subset_name in REMOVE_NEW_LINE else False
        self.extract_question = True if subset_name in EXTRACT_QUESTION else False

    def translate(self, texts):
        user_messages_list = []
        assistant_messages_list = []

        for text in texts:
            if self.remove_new_line:
                text["user"] = text["user"].replace("\n", "")

            if self.extract_question:
                text["user"] = text["user"].split("\n")[0]

            messages = [
                {"role": "user", "content": self.prompt.format(text=text["user"])},
            ]

            user_messages_list.append(messages)

        for text in texts:
            messages = [
                {
                    "role": "user",
                    "content": self.prompt.format(text=text["assistant"]),
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
