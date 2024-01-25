# Input preprocessing and encoding

## Importing Necessary Libraries
- Import libraries and modules like `torch`, `torchvision`, `bindsnet`, and others necessary for data processing and neural network operations.

## Loading Standard MNIST Dataset
- The MNIST dataset is accessed using a wrapper around `torchvision.datasets.MNIST`. This is done in your script with the `MNIST` class from `bindsnet.datasets`.

## Preprocessing the Images
- **Normalization**: The images are first normalized to a range of `[0, 1]` using `transforms.ToTensor()`. This converts the pixel values from `[0, 255]` to `[0, 1]`.
- **Intensity Scaling**: The normalized pixel values are then scaled by a specified intensity factor (e.g., 128) using a lambda function in `transforms.Lambda(lambda x: x * intensity)`. This step is crucial for adjusting the input for the spiking neural network.

## Creating the Dataset with Preprocessing
- An instance of the MNIST dataset is created with the above preprocessing steps. This instance (`train_dataset`) also includes arguments for the Poisson encoder and other dataset parameters like `root`, `download`, and `train`.

## Encoding Images into Spike Trains
- The `PoissonEncoder` from BindsNET is used to encode the preprocessed images into spike trains.
- For each pixel in the image, the encoder generates a series of spikes over time, where the number and timing of spikes are determined based on the pixel's intensity.
- The result is an `encoded_image`, a tensor representing the spike-encoded version of the image. This tensor is significantly larger in size compared to the original image tensor, as it includes temporal information for each pixel.

## Structured Data in `train_dataset`
- Each item in `train_dataset` is a dictionary containing:
  - The original image tensor (`image`).
  - The class label (`label`).
  - The spike-encoded image tensor (`encoded_image`).
  - The encoded label (`encoded_label`), which in this case, remains the same as the original label.

## Ready for Use in Spiking Neural Networks
- With these steps completed, `train_dataset` is now a collection of items where each item has both the original and spike-encoded information, ready to be utilized in spiking neural network models for training and evaluation.

## About Poisson Coding and Firing Rate
- The `PoissonEncoder` simulates a neuron's response to stimulus intensity by generating spikes at a rate proportional to the pixel's intensity after preprocessing.
- The firing rate, measured in Hz (spikes per second), determines the average number of spikes generated. For example, a firing rate of 63.75 Hz means that, on average, 63.75 spikes are expected to occur each second for the maximum intensity pixel after preprocessing.
- This encoding method allows the translation of intensity information into temporal spike patterns, suitable for processing in spiking neural networks.

#===============

# Neuron Label Assignments

Neuron label assignments in a spiking neural network are based on the neurons' response to different types of input during training. The idea is to label each neuron with the class of input to which it responds the most frequently.

### Example
- Imagine a network is trained on a dataset of images, each labeled with a class (like digits 0-9 in MNIST).
- **Neuron A** spikes most often when the network is presented with images of digit '5'.
- **Neuron B** shows the highest frequency of spikes for digit '2'.
- **Assignment**: Consequently, Neuron A is assigned the label '5', and Neuron B is assigned the label '2'.

# Evaluation Methods

### All Activity Accuracy

The "All Activity" method sums up the spike count for each class label and classifies an input based on the label that accumulates the most spikes.

#### Example
- Assume the network is presented with an image of digit '5'.
- **Neuron Response**: Neuron A (labeled '5') fires 20 times, while Neuron B (labeled '2') fires 5 times.
- **Classification**: The input is classified as '5' because the total spike count for '5' (20 spikes) is higher than for '2' (5 spikes).

### Proportion Weighting Accuracy

In the "Proportion Weighting" method, the classification considers both the spike count and the historical firing accuracy of each neuron for different classes. It uses a weighted sum based on these factors.

#### Example
- The network encounters the same image of digit '5'.
- **Historical Proportions**: Based on past training, Neuron A fires for class '5' about 90% of the time, while Neuron B fires for class '2' around 80% of the time.
- **Neuron Response**: In response to the current input, Neuron A fires 20 times, and Neuron B fires 10 times.
- **Weighted Vote Calculation**: 
  - For class '5': \(20 \times 0.90 = 18\)
  - For class '2': \(10 \times 0.80 = 8\)
- **Classification**: The input is classified as '5' because the weighted vote for '5' (18) is higher than for '2' (8).

These methods demonstrate how a spiking neural network uses neuronal firing patterns to classify inputs. While the "All Activity" method provides a straightforward approach, "Proportion Weighting" offers a more nuanced classification by factoring in the reliability and historical accuracy of each neuron's response.
 
