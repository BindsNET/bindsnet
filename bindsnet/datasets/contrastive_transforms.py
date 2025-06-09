import torch
import random
from typing import Optional

def generate_positive_sample(
    x_input: torch.Tensor,
    true_label: int,
    num_classes: int,
) -> torch.Tensor:
    """
    Generates a positive sample x_pos by embedding the true label into x_input.

    The first `num_classes` elements of the output vector are set to 0,
    except at the index corresponding to the `true_label`, where it's
    set to the maximum value of `x_input`. The remaining elements are copied
    from `x_input`.

    Args:
        x_input: The original flattened input vector (1D Tensor).
        true_label: The 0-indexed true class label of x_input.
        num_classes: The total number of classes (c).

    Returns:
        A new tensor x_pos with the true label embedded.
    """
    if not isinstance(x_input, torch.Tensor) or x_input.ndim != 1:
        raise ValueError("x_input must be a 1D PyTorch Tensor.")
    if not (0 <= true_label < num_classes):
        raise ValueError(
            f"True label {true_label} is out of bounds for {num_classes} classes."
        )
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")


    d = x_input.shape[0]
    m = torch.max(x_input) if d > 0 else torch.tensor(0.0, dtype=x_input.dtype) # Handle empty x_input

    # Initialize x_pos with zeros, matching dtype and device of x_input
    x_pos = torch.zeros_like(x_input)

    # Part 1: Embed the true label in the first `num_classes` elements.
    # All these elements are 0, except at the index `true_label`.
    if true_label < min(num_classes, d): # Ensure true_label is within bounds of the modifiable part
        x_pos[true_label] = m

    # Part 2: Copy the rest of the original input vector.
    # These are elements from index `num_classes` to `d-1`.
    if d > num_classes:
        x_pos[num_classes:] = x_input[num_classes:]
    # If num_classes >= d, only the first d elements are modified, and the above copy is skipped.

    return x_pos


def generate_negative_sample(
    x_input: torch.Tensor,
    true_label: int,
    num_classes: int,
    false_label_override: Optional[int] = None,
) -> torch.Tensor:
    """
    Generates a negative sample x_neg by embedding a false label into x_input.

    The first `num_classes` elements of the output vector are set to 0,
    except at the index corresponding to the `chosen_false_label`, where it's
    set to the maximum value of `x_input`. The remaining elements are copied
    from `x_input`.

    Args:
        x_input: The original flattened input vector (1D Tensor).
        true_label: The 0-indexed true class label of x_input.
        num_classes: The total number of classes (c).
        false_label_override: Optional. A specific 0-indexed false class label to embed.
                              If None, a false label will be chosen randomly, ensuring
                              it's different from `true_label`. This parameter can be
                              used if implementing a "hard labeling" strategy externally.
    Returns:
        A new tensor x_neg with the false label embedded.
    """
    if not isinstance(x_input, torch.Tensor) or x_input.ndim != 1:
        raise ValueError("x_input must be a 1D PyTorch Tensor.")
    if not (0 <= true_label < num_classes):
        raise ValueError(
            f"True label {true_label} is out of bounds for {num_classes} classes."
        )
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")

    chosen_false_label: int
    if false_label_override is not None:
        chosen_false_label = false_label_override
        if not (0 <= chosen_false_label < num_classes):
            raise ValueError(
                f"Provided false_label_override {chosen_false_label} is out of bounds for {num_classes} classes."
            )
        if chosen_false_label == true_label:
            raise ValueError(
                f"Provided false_label_override {chosen_false_label} cannot be the same as true_label {true_label}."
            )
    else:
        if num_classes <= 1:
            raise ValueError(
                "Cannot randomly choose a distinct false label with less than 2 classes."
            )
        possible_false_labels = [i for i in range(num_classes) if i != true_label]
        if not possible_false_labels: # Should be caught by num_classes <= 1
             raise ValueError(f"No available false labels to choose from for true_label {true_label} with {num_classes} classes.")
        chosen_false_label = random.choice(possible_false_labels)

    d = x_input.shape[0]
    m = torch.max(x_input) if d > 0 else torch.tensor(0.0, dtype=x_input.dtype) # Handle empty x_input

    # Initialize x_neg with zeros, matching dtype and device of x_input
    x_neg = torch.zeros_like(x_input)

    # Part 1: Embed the false label in the first `num_classes` elements.
    if chosen_false_label < min(num_classes, d): # Ensure chosen_false_label is within bounds
        x_neg[chosen_false_label] = m

    # Part 2: Copy the rest of the original input vector.
    if d > num_classes:
        x_neg[num_classes:] = x_input[num_classes:]

    return x_neg



# --- Example Usage (for demonstration if you run this file directly) ---
if __name__ == "__main__":
    # Example: 10 features in original input, 4 classes
    original_x = torch.rand(10) # Random data
    true_class_label = 1
    total_classes = 4

    print(f"Original x_input: {original_x}")
    print(f"True label: {true_class_label}")
    print(f"Num classes: {total_classes}")
    print("-" * 30)

    x_positive = generate_positive_sample(
        original_x, true_class_label, total_classes
    )
    print(f"x_pos (true label {true_class_label} embedded): {x_positive}")
    print("-" * 30)

    x_negative_random = generate_negative_sample(
        original_x, true_class_label, total_classes
    )
    print(f"x_neg (random false label embedded): {x_negative_random}")
    print("-" * 30)

    specific_false = 3
    if specific_false == true_class_label: # Ensure it's actually false for the example
        specific_false = 0 if true_class_label !=0 else 2

    x_negative_specific = generate_negative_sample(
        original_x, true_class_label, total_classes, false_label_override=specific_false
    )
    print(f"x_neg (specific false label {specific_false} embedded): {x_negative_specific}")
    print("-" * 30)

    # Edge case: num_classes > len(x_input)
    short_x = torch.tensor([0.1, 0.9])
    true_short_label = 0
    classes_short = 3
    print(f"Short Original x_input: {short_x}")
    x_pos_short = generate_positive_sample(short_x, true_short_label, classes_short)
    print(f"x_pos_short (true label {true_short_label}, num_classes {classes_short}): {x_pos_short}")
    x_neg_short = generate_negative_sample(short_x, true_short_label, classes_short, false_label_override=1)
    print(f"x_neg_short (false label 1, num_classes {classes_short}): {x_neg_short}")
