import json


def read_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def transform_data(data):
    labels = []
    bboxs = []
    centers = []

    for key, value in data["labels"].items():
        labels.append(value["label"])
        bboxs.append(value["bbox"])
        centers.append((value["center_x"], value["center_y"]))

    return {
        "labels": labels,
        "bboxs": bboxs,
        "centers": centers,
        "detailed": data["detailed"],
        "id": data["id"],
        "ext": data["ext"],
    }


def save_to_jsonl(data_list, output_file_path):
    with open(output_file_path, "w", encoding="utf-8") as file:
        for data in data_list:
            file.write(json.dumps(data) + "\n")


jsonl_data = read_jsonl("./all_0712.jsonl")

transformed_data_list = [transform_data(data) for data in jsonl_data]

save_to_jsonl(transformed_data_list, "./all_0712_convert.jsonl")
