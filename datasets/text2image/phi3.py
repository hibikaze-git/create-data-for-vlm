import random

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

DESIGN_LIST = [
    "Minimalist website with a lot of whitespace and simple typography.",
    "Dark mode-themed website with high-contrast color scheme.",
    "Mobile-responsive website with large, easily tappable buttons.",
    "A website with an interactive, parallax scrolling effect.",
    "A retro-inspired website with pixel art and vintage fonts.",
    "A website with a custom, hand-drawn illustration theme.",
    "A clean, modern website with bold typography and a monochromatic color scheme.",
    "A website with a nature-inspired theme, using organic shapes and earthy colors.",
    "A website with a strong brand identity, using a custom logo and color palette.",
    "A website designed for storytelling, with a cinematic, narrative-driven layout.",
    "A website with an emphasis on typography, using unique fonts and creative text layouts.",
    "A website with a futuristic, tech-inspired theme, using neon colors and geometric shapes.",
    "A website that incorporates motion graphics and animations.",
    "A website with an immersive, virtual reality experience.",
    "A website that uses minimal imagery, focusing instead on text and typography.",
    "A website with a playful, whimsical theme, using bright colors and cartoonish illustrations.",
    "A website with a luxurious, high-end aesthetic, using rich colors and elegant typography.",
    "A website with a handmade, craft-inspired theme, using textured backgrounds and hand-drawn elements.",
    "A website with a geometric, modular design, using shapes and grids to create a clean layout.",
    "A website with a hand-sketched, pencil-drawn illustration style.",
    "A website that uses bold, eye-catching graphics and illustrations.",
    "A website with a minimalist, no-frills style, focusing on content and functionality.",
    "A website with a vintage, old-school design, using retro fonts and pixel art.",
    "A website with a minimalist color scheme, using only a few complementary colors.",
    "A website with an interactive, game-like interface, incorporating elements of gamification.",
    "A website with a strong emphasis on white space, creating a clean, open layout.",
    "A website with a hand-lettered, calligraphy-inspired typography style.",
    "A website with a hand-painted, watercolor illustration style.",
    "A website with an asymmetrical, unconventional layout, breaking design rules.",
    "A website with a strong, consistent visual hierarchy, guiding the user's eye through the content.",
    "A website with a hand-crafted, rustic aesthetic, using natural materials and textures.",
    "A website with a bold, graphic, poster-style design.",
    "A website with a clean, modern, sans-serif typography style.",
    "A website with an organic, hand-drawn illustration style.",
    "A website with a strong, cohesive brand identity, using consistent imagery and typography.",
    "A website with a minimalist, Bauhaus-inspired design.",
    "A website with a pop art-inspired aesthetic, using bright colors and bold graphics.",
    "A website with a hand-sketched, doodle-style illustration.",
    "A website with a vintage, mid-century modern design, using retro fonts and patterns.",
    "A website with a minimalist, Scandinavian-inspired design, using simple shapes and a muted color palette.",
]


GEN_HTML_PROMPT = """You are a professional web designer.
Create index.html using your imagination while following the instructions below.
1. {design_prompt}
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
            "max_new_tokens": 8192,
            "return_full_text": False,
            "temperature": 0.0,
            "do_sample": False,
        }

    def translate(self, text_batch):
        messages_list = []
        design_prompt = random.choice(DESIGN_LIST)

        for item in text_batch:
            messages = [
                {
                    "role": "user",
                    "content": GEN_HTML_PROMPT.format(
                        design_prompt=design_prompt, text=item["text"]
                    ),
                },
            ]

            messages_list.append(messages)

        outputs = self.pipe(messages_list, **self.generation_args)

        synthesis_dict_list = [
            {
                "html": output[0]["generated_text"].strip(),
                "text": item["text"],
                "id": item["id"],
                "design_prompt": design_prompt,
            }
            for item, output in zip(text_batch, outputs)
        ]

        return synthesis_dict_list
