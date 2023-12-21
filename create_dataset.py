import json

from util import generate_examples, save_img

if __name__ == "__main__":
    NOISE_LEVEL = 0.3
    IMG_SIZE = 100
    NR_OF_EXAMPLES = 1000

    count = 0
    labels = {}

    for img, params in generate_examples(NOISE_LEVEL, IMG_SIZE):
        name = f"img_{count}.png"
        labels[name] = params._asdict()
        save_img(img, "data/" + name)
        count += 1

        if count > NR_OF_EXAMPLES:
            break

    with open("data/dataset.json", "w") as f:
        json.dump(labels, f)
    