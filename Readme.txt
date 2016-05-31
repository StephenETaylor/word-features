 this directory is intended to contain a neural network based on the 
article Bengio, Ducharme, Vincent, Javin: "A Neural Probabilistic Language 
Model" Journal of Machine Learning Research 3(2003)1137-1155

It uses the Emma files in the parent directory for input.  These have a
vocabulary size of 2500, with out-of-vocabulary represented by 0, and all
proper nouns compressed to a single vocabulary item.

Following Bengio, et al, the plan is to use four input words to predict an
output word.  In their notation this is an n=4, or order 4, network.
We'll use 100 features, where the point of the whole exercise is to
learn the features, since we could easily use n-grams to predict the word;

We could consider the 100 features to be a hidden layer, with a one-hot
encoding of the four words in 10000 input nodes, or we could follow the
notation of the article, which (equivalently) places the features in an
array, and indexes into it, thus providing n*100 input features for an
order-n network.  (Or maybe that isn't quite equivalent, since in their
scheme there are coefficients for each feature at each position.)

I will have to stretch to understand how to compute the gradients for back-
propagation in either scheme...

The article uses a hidden layer (after the features vector) of 60 units,
and an output layer equal in size to the vocabulary, estimating probabilities
with a normalized energy, computed by summing the activation energies of 
all units in the output layer, 2500 in our case.

[Is it possible to compute instead an output feature vector, and estimate 
likelihoods based on distances in feature-space? Certainly the computation
requirements seem much gentler...]

So:
Four words of input (I'm centering instead of preceding, this is less useful
for speech recognition, but my goal is to come up with feature vectors)

(100 units/word of features)

60 hidden units

2500 units of output.
I'll start with a clumsy Octave script to evaluate a NN.


