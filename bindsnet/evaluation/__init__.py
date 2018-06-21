import sys
import torch


def assign_labels(spikes, labels, n_labels, rates=None, alpha=1.0):
    '''
    Assign labels to the neurons based on highest average spiking activity.

    Inputs:

        | :code:`spikes` (:code:`torch.Tensor`): Binary tensor of shape
        :code:`(n_samples, time, n_neurons)` of a single layer's spiking activity.
        | :code:`labels` (:code:`torch.Tensor`): Vector of shape :code:`(n_samples,)`
        with data labels corresponding to spiking activity.
        | :code:`n_labels` (:code:`int`): The number of target labels in the data.
        | :code:`rates` (:code:`torch.Tensor`): If passed, these represent spike
        rates from a previous :code:`assign_labels()` call.
        | :code:`alpha` (:code:`float`): Rate of decay of label assignments.

    Returns:

        | (:code:`torch.Tensor`): Vector of shape
        :code:`(n_neurons,)` of neuron label assignments.
        | (:code:`torch.Tensor`): Vector of shape :code:`(n_neurons, n_labels)`
        of proportions of firing activity per neuron, per data label.
    '''
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


def all_activity(spikes, assignments, n_labels):
    '''
    Classify data with the label with highest average spiking activity over all neurons.

    Inputs:

        | :code:`spikes` (:code:`torch.Tensor`): Binary tensor of shape
        :code:`(n_samples, time, n_neurons)` of a layer's spiking activity.
        | :code:`assignments` (:code:`torch.Tensor`): A vector of shape
        :code:`(n_neurons,)` of neuron label assignments.
        | :code:`n_labels` (:code:`int`): The number of target labels in the data.

    Returns:

        | (:code:`torch.Tensor`): Predictions tensor of shape :code:`(n_samples,)`
        resulting from the "all activity" classification scheme.
    '''
    n_samples = spikes.size(0)

    # Sum over time dimension (spike ordering doesn't matter).
    spikes = spikes.sum(1)

    # Summed rates per label (to be calculated).
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
    predictions = torch.sort(rates, dim=1, descending=True)[1][:, 0]

    return predictions


def proportion_weighting(spikes, assignments, proportions, n_labels):
    '''
    Classify data with the label with highest average spiking
    activity over all neurons, weighted by class-wise proportion.

    Inputs:

        | :code:`spikes` (:code:`torch.Tensor`): Binary tensor of shape
        :code:`(n_samples, time, n_neurons)` of a single layer's spiking activity.
        | :code:`assignments` (:code:`torch.Tensor`): A vector of shape
        :code:`(n_neurons,)` of neuron label assignments.
        | :code:`proportions` (torch.Tensor): A matrix of shape :code:`(n_neurons, n_labels)`
        giving the per-class proportions of neuron spiking activity.
        | :code:`n_labels` (:code:`int`): The number of target labels in the data.

    Returns:

        | (:code:`torch.Tensor`): Predictions tensor of shapez:code:`(n_samples,)`
        resulting from the "proportion weighting" classification scheme.
    '''
    n_samples = spikes.size(0)

    # Sum over time dimension (spike ordering doesn't matter).
    spikes = spikes.sum(1)

    # Summed rates per label (to be calculated).
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

def ngram(spikes, ngram_scores, n_labels, n=2) :
    '''
    Predicts between :code:`n_labels` using :code:`ngram_scores`.

    Inputs:

       | :code:`spikes` (:code:`tensor.Tensor`): the spiking activity of shape :code:`(n_ex, n_layer, timesteps)`.
       | :code:`n_labels` (:code:`int`): The number of target labels in the data.
       | :code:`n` (:code:`int`): The max size of ngram to use.
       | :code:`ngram_scores` (:code:`dict`): Previously recorded scores to update.

    Outputs:

       | :code:`prediction` (:code:`list`): Predictions per example. Shape: :code:`(n_ex)`.

        Accuracy
        Confusion Matrix
        Top-k
        Precision
        Recall
    '''
    normalize = lambda x: x/torch.sum(x)

    predictions = []
    for activity in spikes:
       # Initialize score array
       current_score = torch.zeros(n_labels)

       # Obtain firing order for spiking activity
       fire_order = []

       # Keep those timesteps that have firing neurons
       timesteps_to_keep = torch.nonzero(torch.sum(activity, dim=0))
       activity = activity[:, timesteps_to_keep]

       # Aggregate all of the firing neurons' indices
       for timestep in range(activity.size()[1]):
           ordering = [neuronID for pseudo_list in torch.nonzero(activity[:, timestep].view(-1))[:][:].tolist() for neuronID in pseudo_list]
           fire_order += ordering

        # Consider all n_gram subsequences
       for beg in range(len(fire_order)-n+1):
           if tuple(fire_order[beg : beg+n]) in ngram_scores:
               current_score += ngram_scores[tuple(fire_order[beg : beg+n])]

       predictions.append(float(torch.argmax(normalize(current_score))))

    return torch.LongTensor(predictions)

def update_ngram_scores(spikes, gold_labels, n_labels, n=2, ngram_scores={}):
    '''
    Updates ngram scores with n <= n_ngram.

    Inputs:

       | :code:`spikes` (:code:`tensor.Tensor`): the spiking activity of shape :code:`(n_ex, n_layer, timesteps)`.
       | :code:`gold_labels` (:code:`int`) : The ground truth value of shape :code:`(n_ex,)`
       | :code:`n_labels` (:code:`int`): The number of target labels in the data.
       | :code:`n` (:code:`int`): The max size of ngram to use.
       | :code:`ngram_scores` (:code:`dict`): Previously recorded scores to update.

    Outputs:

        | :code:`ngram_scores` (:code:`dict`): Keys are ngram :code:`tuple` and values are
        :code:`torch.Tensor` of size :code:`n_labels` containing scores per label.
    '''

    assert spikes.size()[0] == len(gold_labels), 'Every example must have a golden label'

    for n_ex, activity in enumerate(spikes):
       # Obtain firing order for spiking activity
       fire_order = []
       # Keep those timesteps that have firing neurons
       timesteps_to_keep = torch.nonzero(torch.sum(activity, dim=0))
       activity = activity[:, timesteps_to_keep]

       # Aggregate all of the firing neurons' indices
       for timestep in range(activity.size()[1]):
           ordering = [neuronID for pseudo_list in torch.nonzero(activity[:, timestep].view(-1))[:][:].tolist() for neuronID in pseudo_list]
           fire_order += ordering

       # Add counts for every n-gram
       for i in range(1, n+1):
           for beg in range(len(fire_order)-i+1):
            # For every ordering based on n (i)
            if tuple(fire_order[beg : beg+i]) not in ngram_scores:
                ngram_scores[tuple(fire_order[beg : beg+i])] = torch.zeros(n_labels)
            ngram_scores[tuple(fire_order[beg : beg+i])][int(gold_labels[n_ex])] += 1

    return ngram_scores

