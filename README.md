# Circle Detection using a CNN

## Dataset

The datasets for training, testing, and evaluating the model can be created using the `create_dataset.py` script. Each dataset consists of square, grayscale, blurry images featuring a white circle with an arbitrary radius. Users can set the size of the square images and the noise level.

```bash
python3 create_dataset.py --img_size IMG_SIZE --noise_level NOISE_LEVEL
```

## Model

The model architecture is a standard Convolutional Neural Network (CNN). It features three convolutional layers with increasing channel depth (32, 64, 128), each followed by batch normalization and max pooling. This enhances feature extraction and reduces overfitting. The chosen activation function is ReLU, applicable to both the convolutional and fully connected layers. Dropout layers are incorporated after the fully connected layers to mitigate overfitting further. This architecture is particularly effective for blurry images, capturing subtle, spread-out features

## Training

The model, specified in `model/cnn.py`, can be trained using `train.py`. The relevant training parameters, such as the learning rate and the batch size, can be set in a JSON file. The following parameters were used for training and have to be defined for the training to run:

```
"batch_size": 64,
"lr": 0.0001,
"kernel_size": 3,
"num_epochs": 50,
"save_every": 20
```
After 40 epochs, the learning rate is reduced to 10% of the original value. The loss function is a custom combination of MSE and L1 loss, defined as:

$$
L = \frac{1}{n} \sum_{i=1}^{n} (r_{i} - \hat{r}_{i})^2 + (c{i} - \hat{c}_{i})^2 + |\rho_{i} - \hat{\rho}_{i}|
$$
Where values with a hat denote predicted values, and $r$, $c$, $\rho$ denote the row, column, and radius of the circle, respectively. This custom loss function is defined in `loss.py`.

 Training can then be run with the following command:

```bash
python3 train.py --data_path data/ --config_file config/parameter.json
```
Training utilizes the GPU if CUDA is available.

## Results

The loss on the train and test set, as well as the Mean IoU of the predicted circles, is displayed in the following two plots:

<div>
  <img src="plots/loss.pdf" alt="Loss" style="display: inline-block;">
  <img src="plots/iou.pdf" alt="Mean IoU" style="display: inline-block;">
</div>

The results on the held-out validation set are displayed in the following table. Accuracy denotes the thresholded accuracy, where a predicted is considered correct when the IoU score of the prediction is above the threshold.

| Metric                | Value     |
|-----------------------|-----------|
| Accuracy (Threshold 0.5) | 0.75   |
| Accuracy (Threshold 0.6) | 0.59   |
| Accuracy (Threshold 0.7) | 0.39   |
| Mean IoU              | 0.62      |

- The accuracy and Mean IoU indicate moderate performance, especially at lower thresholds. However, the performance drop at higher thresholds suggests difficulties in precise circle detection.
