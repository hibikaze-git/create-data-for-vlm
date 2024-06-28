from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

GEN_HTML_PROMPT = """You are a professional web designer.
Create index.html using your imagination while following the instructions below.
1. A website with a vintage, old-school design, using retro fonts and pixel art.
2. Use a high-quality design.
3. Include fonts of various sizes, large and small.
4. Use a variety of fonts.
5. Make sure the design is not too simple.
6. Display the following Japanese text. Do not include any other text. Do not omit any Japanese text, display the entire text.
{text}
7. Answer with html code only"""


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
        )

        self.generation_args = {
            "max_new_tokens": 8192,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }

    def translate(self, text):
        messages_list = []

        messages = [
            {"role": "user", "content": GEN_HTML_PROMPT.format(text=text)},
        ]

        messages_list.append(messages)

        outputs = self.pipe(messages_list, **self.generation_args)

        synthesis_dict = {"html": outputs[0][0]["generated_text"].strip()}

        return synthesis_dict
