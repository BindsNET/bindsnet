from itertools import product

import torch

from typing import Optional, Tuple, Dict


def assign_labels(spikes: torch.Tensor, labels: torch.Tensor, n_labels: int, rates: Optional[torch.Tensor] = None,
                  alpha: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # language=rst
    """
    Assign labels to the neurons based on highest average spiking activity.

    :param spikes: Binary tensor of shape ``(n_samples, time, n_neurons)`` of a single layer's spiking activity.
    :param labels: Vector of shape ``(n_samples,)`` with data labels corresponding to spiking activity.
    :param n_labels: The number of target labels in the data.
    :param rates: If passed, these represent spike rates from a previous ``assign_labels()`` call.
    :param alpha: Rate of decay of label assignments.
    :return: Tuple of class assignments, per-class spike proportions, and per-class firing rates.
    """
    n_neurons = spikes.size(2)

    if rates is None:
        rates = torch.zeros_like(torch.Tensor(n_neurons, n_labels))

    # Sum over time dimension (spike ordering doesn't matter).
    spikes = spikes.sum(1)

    for i in range(n_labels):
        # Count the number of samples with this label.
        n_labeled = torch.sum(labels == i).float()

        if n_labeled > 0:
            # Get indices of samples with this label.
            indices = torch.nonzero(labels == i).view(-1)

            # Compute average firing rates for this label.
            rates[:, i] = alpha * rates[:, i] + (torch.sum(spikes[indices], 0) / n_labeled)

    # Compute proportions of spike activity per class.
    proportions = rates / rates.sum(1, keepdim=True)
    proportions[proportions != proportions] = 0  # Set NaNs to 0

    # Neuron assignments are the labels they fire most for.
    assignments = torch.max(proportions, 1)[1]

    return assignments, proportions, rates


def all_activity(spikes: torch.Tensor, assignments: torch.Tensor, n_labels: int) -> torch.Tensor:
    # language=rst
    """
    Classify data with the label with highest average spiking activity over all neurons.

    :param spikes: Binary tensor of shape ``(n_samples, time, n_neurons)`` of a layer's spiking activity.
    :param assignments: A vector of shape ``(n_neurons,)`` of neuron label assignments.
    :param n_labels: The number of target labels in the data.
    :return: Predictions tensor of shape ``(n_samples,)`` resulting from the "all activity" classification scheme.
    """
    n_samples = spikes.size(0)

    # Sum over time dimension (spike ordering doesn't matter).
    spikes = spikes.sum(1)

    rates = torch.zeros(n_samples, n_labels)
    for i in range(n_labels):
        # Count the number of neurons with this label assignment.
        n_assigns = torch.sum(assignments == i).float()

        if n_assigns > 0:
            # Get indices of samples with this label.
            indices = torch.nonzero(assignments == i).view(-1)

            # Compute layer-wise firing rate for this label.
            rates[:, i] = torch.sum(spikes[:, indices], 1) / n_assigns

    # Predictions are arg-max of layer-wise firing rates.
    return torch.sort(rates, dim=1, descending=True)[1][:, 0]


def proportion_weighting(spikes: torch.Tensor, assignments: torch.Tensor, proportions: torch.Tensor,
                         n_labels: int) -> torch.Tensor:
    # language=rst
    """
    Classify data with the label with highest average spiking activity over all neurons, weighted by class-wise
    proportion.

    :param spikes: Binary tensor of shape ``(n_samples, time, n_neurons)`` of a single layer's spiking activity.
    :param assignments: A vector of shape ``(n_neurons,)`` of neuron label assignments.
    :param proportions: A matrix of shape ``(n_neurons, n_labels)`` giving the per-class proportions of neuron spiking
                        activity.
    :param n_labels: The number of target labels in the data.
    :return: Predictions tensor of shape ``(n_samples,)`` resulting from the "proportion weighting" classification
             scheme.
    """
    n_samples = spikes.size(0)

    # Sum over time dimension (spike ordering doesn't matter).
    spikes = spikes.sum(1)

    rates = torch.zeros(n_samples, n_labels)
    for i in range(n_labels):
        # Count the number of neurons with this label assignment.
        n_assigns = torch.sum(assignments == i).float()

        if n_assigns > 0:
            # Get indices of samples with this label.
            indices = torch.nonzero(assignments == i).view(-1)

            # Compute layer-wise firing rate for this label.
            rates[:, i] += torch.sum((proportions[:, i] * spikes)[:, indices], 1) / n_assigns

    # Predictions are arg-max of layer-wise firing rates.
    predictions = torch.sort(rates, dim=1, descending=True)[1][:, 0]

    return predictions


def ngram(spikes: torch.Tensor, ngram_scores: Dict[Tuple[int, ...], torch.Tensor], n_labels: int,
          n: int) -> torch.Tensor:
    # language=rst
    """
    Predicts between ``n_labels`` using ``ngram_scores``.

    :param spikes: Spikes of shape ``(n_examples, time, n_neurons)``.
    :param ngram_scores: Previously recorded scores to update.
    :param n_labels: The number of target labels in the data.
    :param n: The max size of n-gram to use.
    :return: Predictions per example.
    """
    predictions = []
    for activity in spikes:
        score = torch.zeros(n_labels)

        # Aggregate all of the firing neurons' indices
        fire_order = []
        for t in range(activity.size()[0]):
            ordering = torch.nonzero(activity[t].view(-1))
            if ordering.numel() > 0:
                fire_order += ordering[:, 0].tolist()

        # Consider all n-gram sequences.
        for j in range(len(fire_order) - n):
            if tuple(fire_order[j:j + n]) in ngram_scores:
                score += ngram_scores[tuple(fire_order[j:j + n])]

        predictions.append(torch.argmax(score))

    return torch.LongTensor(predictions)


def update_ngram_scores(spikes: torch.Tensor, labels: torch.Tensor, n_labels: int, n: int,
                        ngram_scores: Dict[Tuple[int, ...], torch.Tensor]) -> Dict[Tuple[int, ...], torch.Tensor]:
    # language=rst
    """
    Updates ngram scores by adding the count of each spike sequence of length n from the past ``n_examples``.

    :param spikes: Spikes of shape ``(n_examples, time, n_neurons)``.
    :param labels: The ground truth labels of shape ``(n_examples)``.
    :param n_labels: The number of target labels in the data.
    :param n: The max size of n-gram to use.
    :param ngram_scores: Previously recorded scores to update.
    :return: Dictionary mapping n-grams to vectors of per-class spike counts.
    """
    for i, activity in enumerate(spikes):
        # Obtain firing order for spiking activity.
        fire_order = []

        # Aggregate all of the firing neurons' indices.
        for t in range(spikes.size(1)):
            # Gets the indices of the neurons which fired on this timestep.
            ordering = torch.nonzero(activity[t]).view(-1)
            if ordering.numel() > 0:  # If there was more than one spike...
                # Add the indices of spiked neurons to the fire ordering.
                ordering = ordering.tolist()
                fire_order.append(ordering)

        # Check every sequence of length n.
        for order in zip(*(fire_order[k:] for k in range(n))):
            for sequence in product(*order):
                if sequence not in ngram_scores:
                    ngram_scores[sequence] = torch.zeros(n_labels)

                ngram_scores[sequence][int(labels[i])] += 1

    return ngram_scores
