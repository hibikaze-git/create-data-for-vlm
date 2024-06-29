from html2image import Html2Image

from datasets import load_dataset

hti = Html2Image(output_path="./images")

dataset = load_dataset("team-hatakeyama-phase2/text2html", cache_dir="./cache")


def parse_html(html_text):
    html_text = "<!DOCTYPE html>" + html_text.split("<!DOCTYPE html>")[-1]
    html_text = html_text.split("</html>")[0] + "</html>"
    html_text = html_text.replace('class="animated-text"', "")

    return html_text


for i, data in enumerate(dataset["train"]):
    html_text = data["html"]

    if (
        "<!DOCTYPE html>" in html_text
        #and '<html lang="ja">' in html_text
        and "</html>" in html_text
    ):
        html_text = parse_html(html_text)

        with open("./index.html", "w", encoding="utf-8") as f:
            f.write(html_text)

        hti.screenshot(html_file="./index.html", save_as=f"{i}.jpg")

    if i == 99:
        break
