import json
import os

from util import generate_examples, save_img

if __name__ == "__main__":
    NOISE_LEVEL = 0.3
    IMG_SIZE = 100
    NR_OF_EXAMPLES = [5000, 500, 500]
    SET_NAMES = ["train", "test", "val"]

    for i, set_name in enumerate(SET_NAMES):

        data_dir = f"data/{set_name}"

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        count = 0
        labels = {}

        for img, params in generate_examples(NOISE_LEVEL, IMG_SIZE):
            name = f"img_{count}.png"
            labels[name] = params._asdict()
            img_path = f"{data_dir}/{name}"
            save_img(img, img_path)
            count += 1

            if count > NR_OF_EXAMPLES[i]:
                break

        with open(f"{data_dir}/dataset.json", "w") as f:
            json.dump(labels, f)
