import torch
import random
from typing import Optional

def prepend_label_to_image(
    x_input: torch.Tensor,
    label: int,
    num_classes: int,
) -> torch.Tensor:
    """
    Generates a sample by embedding a label into x_input.

    The first `num_classes` elements of the output vector are set to 0,
    except at the index corresponding to the `label`, where it's
    set to the maximum value of `x_input`. The remaining elements are copied
    from `x_input` starting after the label section.

    Args:
        x_input: The original flattened input vector (1D Tensor).
        label: The 0-indexed class label to embed.
        num_classes: The total number of classes (c).

    Returns:
        A new tensor with the label embedded (SAME SIZE as input).
    """
    if not isinstance(x_input, torch.Tensor) or x_input.ndim != 1:
        raise ValueError("x_input must be a 1D PyTorch Tensor.")
    if not (0 <= label < num_classes):
        raise ValueError(
            f"Label {label} is out of bounds for {num_classes} classes."
        )
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")
    
    d = x_input.shape[0]
    if num_classes > d:
        raise ValueError(
            f"num_classes ({num_classes}) cannot be larger than input size ({d})"
        )

    m = torch.max(x_input) if d > 0 else torch.tensor(0.0, dtype=x_input.dtype)

    # FIX: Create output tensor with SAME SIZE as input (not larger)
    x_output = torch.zeros_like(x_input)  # Same size as x_input

    # Part 1: Embed the label in the first `num_classes` elements
    x_output[label] = m

    # Part 2: Copy the remaining original input elements (skip first num_classes)
    if d > num_classes:
        x_output[num_classes:] = x_input[num_classes:]

    return x_output


# --- Example Usage (for demonstration if you run this file directly) ---
if __name__ == "__main__":
    # Example: 10 features in original input, 4 classes
    original_x = torch.rand(10)  # Random data
    true_class_label = 1
    total_classes = 4

    print(f"Original x_input: {original_x}")
    print(f"Original shape: {original_x.shape}")
    print(f"True label: {true_class_label}")
    print(f"Num classes: {total_classes}")
    print("-" * 50)

    # Positive sample (true label)
    x_positive = prepend_label_to_image(
        original_x, true_class_label, total_classes
    )
    print(f"x_pos (true label {true_class_label} embedded): {x_positive}")
    print(f"x_pos shape: {x_positive.shape}")  # Should be [10] - same as input
    print("-" * 50)

    # Negative sample (random false label)
    available_labels = [i for i in range(total_classes) if i != true_class_label]
    rand_negative_label = random.choice(available_labels)
    
    x_negative_random = prepend_label_to_image(
        original_x, rand_negative_label, total_classes
    )
    print(f"x_neg (random false label {rand_negative_label} embedded): {x_negative_random}")
    print(f"x_neg shape: {x_negative_random.shape}")  # Should be [10] - same as input
    print("-" * 50)

    # Test with MNIST-like data
    mnist_like = torch.rand(784)  # Like MNIST flattened
    num_classes_mnist = 5
    
    print(f"MNIST-like original: shape {mnist_like.shape}")
    x_mnist_pos = prepend_label_to_image(mnist_like, 2, num_classes_mnist)
    print(f"MNIST-like after embedding label 2: shape {x_mnist_pos.shape}")  # Should be [784]
    print(f"First 10 elements: {x_mnist_pos[:10]}")
    print(f"Label embedded at position 2: {x_mnist_pos[2]}")  # Should be max value
    print(f"Positions 0,1,3,4 should be 0: {x_mnist_pos[[0,1,3,4]]}")