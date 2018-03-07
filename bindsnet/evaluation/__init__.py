import numpy as np
import torch
'''
Most important functions to use:
confidence_weighting()
ngram()

'''

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
