import numpy as np
import torch

def assign_labels(spikes, labels, n_labels, rates=None, alpha=0.8):
    '''
    Given a sequence of recorded spikes and corresponding labels, assign
        labels to the neurons based on highest average spiking activity.
    
    Inputs:
        spikes (torch.Tensor or torch.cuda.Tensor): Binary tensor of shape
            (n_samples, time, n_neurons) of a single layer's spiking activity.
        labels (torch.Tensor or torch.cuda.Tensor): Vector of shape
            (n_samples,) with data labels corresponding to spiking activity.
        n_labels (int): The number of target labels in the data.
        rates (torch.Tensor or torch.cuda.Tensor): If passed, these
            represent spike rates from a previous assign_labels() call.
    
    Returns:
        (torch.Tensor or torch.cuda.Tensor): A vector of
            shape (n_neurons,) of neuron label assignments.
        (torch.Tensor or torch.cuda.Tensor): A vector of shape (n_neurons, n_labels)
            of proportions of firing activity per neuron, per data label.
    '''
    n_neurons = spikes.size(2) # TODO: make generic. size(-1)?
    
    if rates is None:
        rates = torch.zeros_like(torch.Tensor(n_neurons, n_labels))
    
    # Sum over time dimension (spike ordering doesn't matter).
    spikes = spikes.sum(1)
    
    for i in range(n_labels):
        # Count the number of samples with this label.
        n_labeled = torch.sum(labels == i)
        
        if n_labeled > 0:
            # Get indices of samples with this label.
            indices = torch.nonzero(labels == i).view(-1)
            
            # Compute average firing rates for this label.
            rates[:, i] = alpha * rates[:, i] + (1 - alpha) * torch.sum(spikes[indices], 0) / n_labeled
    
    # Compute proportions of spike activity per class.
    proportions = rates / rates.sum(1, keepdim=True)
    
    # Neuron assignments are the labels they fire most for.
    assignments = torch.max(proportions, 1)[1]

    # Set nans to 0
    proportions[proportions!=proportions] = 0

    return assignments, proportions, rates


def all_activity(spikes, assignments, n_labels):
    '''
    Given neuron assignments and the network spiking activity, new
        data is classified with the label giving the highest average
        spiking activity over all neurons with the label assignment.
    
    Inputs:
        spikes (torch.Tensor or torch.cuda.Tensor): Binary tensor of shape
            (n_samples, time, n_neurons) of a single layer's spiking activity.
        assignments (torch.Tensor or torch.cuda.Tensor): A vector
            of shape (n_neurons,) of neuron label assignments.
        n_labels (int): The number of target labels in the data.
    
    Returns:
        (torch.Tensor or torch.cuda.Tensor): Predictions tensor of shape
            (n_samples,) resulting from the "all activity" classification
            strategy.
    '''
    n_samples = spikes.size(0)
    
    # Sum over time dimension (spike ordering doesn't matter).
    spikes = spikes.sum(1)
    
    # Summed rates per label (to be calculated).
    rates = torch.zeros(n_samples, n_labels)
    
    for i in range(n_labels):
        # Count the number of neurons with this label assignment.
        n_assigns = torch.sum(assignments == i)
        
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
    Given neuron assignments and the network spiking activity, new
        data is classified with the label giving the highest average
        spiking activity over all neurons with the label assignment,
        weighted by the class-wise proportion of spiking activity.
    
    Inputs:
        spikes (torch.Tensor or torch.cuda.Tensor): Binary tensor of shape
            (n_samples, time, n_neurons) of a single layer's spiking activity.
        assignments (torch.Tensor or torch.cuda.Tensor): A vector
            of shape (n_neurons,) of neuron label assignments.
        proportions (torch.Tensor or torch.cuda.Tensor): A matrix of shape
            (n_neurons, n_labels) giving the per-class proportions of neuron
            spiking activity.
        n_labels (int): The number of target labels in the data.
    
    Returns:
        (torch.Tensor or torch.cuda.Tensor): Predictions tensor of shape
            (n_samples,) resulting from the "proportion weighting"
            classification strategy.
    '''
    n_samples = spikes.size(0)
    
    # Sum over time dimension (spike ordering doesn't matter).
    spikes = spikes.sum(1)
    
    # Summed rates per label (to be calculated).
    rates = torch.zeros(n_samples, n_labels)
    
    for i in range(n_labels):
        # Count the number of neurons with this label assignment.
        n_assigns = torch.sum(assignments == i)
        
        if n_assigns > 0:
            # Get indices of samples with this label.
            indices = torch.nonzero(assignments == i).view(-1)
            
            # Compute layer-wise firing rate for this label.
            rates[:, i] += torch.sum((proportions[:, i] * spikes)[:, indices], 1) / n_assigns
    
    # Predictions are arg-max of layer-wise firing rates.
    predictions = torch.sort(rates, dim=1, descending=True)[1][:, 0]
    
    return predictions


def get_fire_order(example):
    # Example.shape = (n_layer, time steps)     <class 'torch.FloatTensor'> torch.Size([100, 350])
    n_layer, timesteps = example.shape
    fire_order = []
    for timestep in range(timesteps): # timeslice.shape is (n_layer)
        if torch.sum(example[:,timestep])>0:
            for n_id in range(n_layer):
                if example[n_id][timestep]:
                    fire_order.append(n_id)
    return fire_order

def normalize_probability(v):
    # v is a numpy array
    return v/np.sum(v)


def ngram(spikes, true_labels, ngrams, n_ngram):
    '''
    Evaluates the network using the confidence weighting scheme.
    Usage: First, call estimate_ngram_probabilities(), and use the ngrams dict returned as input to this function.
    Inputs:
        spikes: the network activity of the last layer, as returned by network.run()
                  shape = (n_samples, n_layer, time steps)
        true_labels: The ground truth values to compare to
                    shape = (n_samples,)
    
    Outputs: 
        Accuracy
        Confusion Matrix
        Top-k
        Precision
        Recall
    '''
    predictions = []
    for example in spikes: # example.shape is (n_layer, time steps)
        ngram_score = np.zeros(10)
        fire_order = get_fire_order(example)

        for i in range(len(fire_order)-n_ngram+1): # Consider all n_gram subsequences
            if tuple(fire_order[i:i+n_ngram]) in ngrams:
                ngram_score += normalize_probability(ngrams[tuple(fire_order[i:i+n_ngram])])

        predictions.append(np.argmax(ngram_score))

    # Now compare to true labels
    accuracy = np.mean([a==b for a,b in zip(predictions,true_labels)])
    return accuracy


def increment_count_ngram(ngrams, seq, true_label):
    if not tuple(seq) in ngrams:
        ngrams[tuple(seq)] = np.zeros(10)
    ngrams[tuple(seq)][true_label] += 1

def estimate_ngram_probabilities(spikes, true_labels, n_ngram):
    '''
    Estimates all ngram probabilities with n<=n_ngram
    Inputs:
        spikes: the network activity of the last layer, as returned by network.run()
                  shape = (n_samples, time steps, n_layer)
        true_labels: The ground truth values to compare to
                    shape = (n_samples,)
        n_ngram: int indicating the max size of ngrams to use
    Outputs:
        ngrams: dict with keys as ngram tuples and values = np.array(int) of shape 10
        containing scores per class
    '''
    ngrams = {}
    for idx,example in enumerate(spikes): # example.shape is (time steps, n_layer)
        fire_order = get_fire_order(example)
        # Add to counts for every n-gram
        for i in range(1,n_ngram+1): # from 1 to n_ngram
            for beg in range(len(fire_order)-i+1): 
                increment_count_ngram(ngrams, fire_order[beg:beg+i], int(true_labels[idx]))
    return ngrams
