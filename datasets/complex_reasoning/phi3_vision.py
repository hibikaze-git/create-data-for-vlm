from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

QUESTION_GEN_PRONPT = """Think of one question about the image that you can infer from its content. Make it a more complex question than simply describing the scene."""


class Phi3VisionManager:
    def __init__(self) -> None:
        model_id = "microsoft/Phi-3-vision-128k-instruct"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="cuda",
            trust_remote_code=True,
            torch_dtype="auto",
            _attn_implementation="flash_attention_2",
            cache_dir="../../cache",
        )  # use _attn_implementation='eager' to disable flash attention

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        self.generation_args = {
            "max_new_tokens": 4096,
            "temperature": 0.0,
            "do_sample": False,
        }

    def gen_text(self, image, prompt):
        messages = [
            {"role": "user", "content": f"<|image_1|>\n{prompt}"},
        ]

        prompt = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(prompt, [image], return_tensors="pt").to("cuda")

        generate_ids = self.model.generate(
            **inputs,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            **self.generation_args,
        )

        generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
        response = self.processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return response.strip()

    def synthesis(self, image_path):
        image = Image.open(image_path)

        synthesis_dict = {}

        # 質問生成
        question = self.gen_text(image, QUESTION_GEN_PRONPT)
        synthesis_dict["question"] = question

        # 回答生成
        synthesis_dict["answer"] = self.gen_text(image, question)

        return synthesis_dict
