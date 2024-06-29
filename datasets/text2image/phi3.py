from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

GEN_HTML_PROMPT = """You are a professional web designer.
Create index.html by following the instructions below.
1. Make sure the design and layout are not too simple.
2. Use colors and shapes effectively.
3. No animation or javascript used.
4. Include the following text. Omission of any text is prohibited: {text}
5. Do not display any text other than the specified text.

Answer with html code only."""


class Phi3Manager:
    def __init__(self) -> None:
        model_id = "microsoft/Phi-3-medium-128k-instruct"
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
            batch_size=4,
        )

        self.generation_args = {
            "max_new_tokens": 8192,
            "return_full_text": False,
            "temperature": 0.7,
            "do_sample": True,
        }

    def translate(self, text_batch):
        messages_list = []

        for item in text_batch:
            messages = [
                {
                    "role": "user",
                    "content": GEN_HTML_PROMPT.format(text=item["text"]),
                },
            ]

            messages_list.append(messages)

        outputs = self.pipe(messages_list, **self.generation_args)

        synthesis_dict_list = [
            {
                "html": output[0]["generated_text"].strip(),
                "text": item["text"],
                "id": item["id"],
            }
            for item, output in zip(text_batch, outputs)
        ]

        return synthesis_dict_list
