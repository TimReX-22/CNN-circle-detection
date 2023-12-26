import json
import os
import argparse

from util import generate_examples, save_img


def parse_cmd_line() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_size", required=True)
    parser.add_argument("--noise_level", required=True)
    return parser.parse_args()


def create_dataset(args: argparse.Namespace):
    num_examples = [5000, 500, 500]
    set_names = ["train", "test", "val"]
    img_size = int(args.img_size)
    noise_level = float(args.noise_level)

    for i, set_name in enumerate(set_names):

        data_dir = f"data/{set_name}"

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        count = 0
        labels = {}

        for img, params in generate_examples(noise_level, img_size):
            name = f"img_{count}.png"
            labels[name] = params._asdict()
            img_path = f"{data_dir}/{name}"
            save_img(img, img_path)
            count += 1

            if count > num_examples[i]:
                break

        with open(f"{data_dir}/dataset.json", "w") as f:
            json.dump(labels, f)


if __name__ == "__main__":
    args = parse_cmd_line()
    create_dataset(args)
