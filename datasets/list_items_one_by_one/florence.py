import os

import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


class FlorenceManager:
    def __init__(self, image_output_dir="./images") -> None:
        model_id = "microsoft/Florence-2-large"

        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                cache_dir="../../cache",
            )
            .eval()
            .cuda()
        )

        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True, cache_dir="../../cache"
        )

        self.image_output_dir = image_output_dir

    def run_example(self, task_prompt, image, text_input=None):
        if text_input is None:
            prompt = task_prompt
        else:
            prompt = task_prompt + text_input

        inputs = self.processor(text=prompt, images=image, return_tensors="pt")

        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"].cuda(),
            pixel_values=inputs["pixel_values"].cuda(),
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )

        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        parsed_answer = self.processor.post_process_generation(
            generated_text, task=task_prompt, image_size=(image.width, image.height)
        )

        return parsed_answer

    def plot_bbox(self, image, data, image_name):
        fig, ax = plt.subplots()
        ax.imshow(image)

        label_positions = []

        idx = 0

        idx_label_dict = {}
        skipped_labels = []

        for i, (bbox, label) in enumerate(zip(data["bboxes"], data["labels"])):
            x1, y1, x2, y2 = bbox

            center_x = x1 + (x2 - x1) / 2
            center_y = y1 + (y2 - y1) / 2

            # ラベルの位置が著しく近いものは除去
            overlap = False
            for lx, ly in label_positions:
                if abs(center_x - lx) < 10 and abs(center_y - ly) < 10:
                    overlap = True
                    break

            obj_info = {
                "label": label,
                "bbox": bbox,
                "center_x": center_x,
                "center_y": center_y,
            }

            # ラベル
            if not overlap:
                # bounding box
                # rect = patches.Rectangle(
                #    (x1, y1), x2 - x1, y2 - y1, linewidth=0.4, edgecolor="black", facecolor="none", alpha=0.9
                # )

                # ax.add_patch(rect)

                plt.text(
                    center_x,
                    center_y,
                    str(idx),
                    color="white",
                    fontsize=7,
                    ha="center",
                    va="center",
                    bbox=dict(facecolor="black", alpha=0.35),
                )

                label_positions.append((center_x, center_y))
                idx_label_dict[idx] = obj_info

                idx += 1
            else:
                skipped_labels.append(obj_info)

        ax.axis("off")

        save_path = os.path.join(self.image_output_dir, f"{image_name}.jpg")

        plt.savefig(save_path, format="jpg", bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        return idx_label_dict, skipped_labels

    def synthesis(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image_name = os.path.splitext(os.path.basename(image_path))[0]

        synthesis_dict = {}

        # caption生成
        task_prompt = "<MORE_DETAILED_CAPTION>"
        # task_prompt = '<DENSE_REGION_CAPTION>'
        results = self.run_example(task_prompt, image)

        # grounding
        text_input = results[task_prompt]
        task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
        results = self.run_example(task_prompt, image, text_input)

        # 画像を保存
        idx_label_dict, skipped_labels = self.plot_bbox(
            image, results["<CAPTION_TO_PHRASE_GROUNDING>"], image_name
        )
        # self.plot_bbox(image, results['<DENSE_REGION_CAPTION>'], image_name)
        # print(text_input)
        # print(idx_label_dict)

        synthesis_dict["labels"] = idx_label_dict
        synthesis_dict["skipped_labels"] = skipped_labels
        synthesis_dict["detailed"] = text_input.strip()
        synthesis_dict["id"] = image_name
        synthesis_dict["ext"] = "jpg"

        return synthesis_dict


if __name__ == "__main__":
    florence_mgr = FlorenceManager()
    synthesis_dict = florence_mgr.synthesis("./000000111118.jpg")
    print(synthesis_dict)
