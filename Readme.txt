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

[several iterations, none of which seems to do much except iterate to equal
 feature values, which then seem to grow.
  iteration 1, hopefully checked into git:
     uses features as output.  unfortunately, any set of features which are
     identical for all words lead to a minimum objective function of zero.
     in addition, the features seem to have a steady growth...
  iteration 2: definitely checked into git:
     uses probability of word (top 100 words only) as an output.  No 
     normalization.  Mysteriously doesn't work all that well.  same 
     problems as before, even though identical feature vectors should converge
     to minimum objective function of one. 
  iteration 3: 
     Modified variable names to match the Bengio paper, and modified the
     computation of output to exponentiate y and normalize to a probability.
     I'm not sure I understand the point, except to introduce non-linearity.
     still has the same convergence problems, I think, except that the
     exponentiation turns grow of y into a numeric overflow almost immediately,
     on the send pass through the data.
]

June 5, 2016
     The paper describes a fix for the numeric overflow, but I thought I'd 
     try regularization first.  I normalize H, U, and C, although there seems
     so be a hint that C shouldn't be normalized, there also is a hint that it 
     is.  (C is the features array.)

I put in the described regularization code.  For R=0 (no regularization) the
objective function seems to grow by about 300 per 10000 stochastic steps.
Thats about 30 per step.  The learning rate is 0.001, so it's hard to see how
it does it!

For R=1, it looks like there is actually some training going on, although
I'm not quite sure.  L continues to increase steadily, at epoch 6 about 6 units
per 10000 steps.  The values of C march up slowly; they are all near -0.027 
but at epoch 9 seem to increase in magnitude by about 11e-6 in per epoch.

For R=2, I sometimes see differences between rows 1 and 2, or especially 99 and
100, but no variation in the columns.  For several epochs before seventeen,
we held at 0.090333, but then began to advance again, at 5E6 per epoch.
Finished twentieth epoch with C weights all 0.090349.
L continues to advance also, finishing at about 6000.

For R=4, L drops during epoch 1 from 2658 after the first 10000 steps, 
to 236.1 after 1st epoch, with all C feature weights at 0.012563.
But in second epoch, it begins to advance again, with C feature weights at 0.012568, and L at 322.99 at the end of epoch 2.
Reaches 1883 at the end of 20 epochs.  Final weights are 0.012650, a steady but
slightly smaller increase.

For R=10, L started at ~ 2e4 and built to 3e4 over 20 epochs; at epoch 9 the 
C features coefficients were all 0.10412; at epoch 20 they were 0.10418
The total # of C's = 2546 * 100, approximately 2.5e5; # of U's = 400 * 60 = 2.4e4; #of H's = 60 * 100 = 6e3, total # of coefficients ~ 2.8e5, contribution of
Regularization term to L ~ 3e4/10/2.8e5 ~ 1e-2, which is approximately 0.1**2
Since we expect the actual error term to be only 1, not so surprising that
the regularization portion dominates.


SO: can we divide the objective function into pieces to report on?  What is 
the  max and min of y? what fraction is for the regularization?  Was there any
opportunity to see singular values of y?

Did I do the back-prop correctly, and does the value of the objective change
from rep to rep?  
  1) although the paper says the objective is to maximize the per-word entropy
     (it doesn't mention the entropy, but I think that's the formula in 
     use)
     a) we want to minimize the per-word entropy anyway
     b) the partial derivatives used are for sum-squared-error.
        * but they are derived as though y, and not p, were the output.
        * since the logs of the probabilities should include one zero and 
          99 minus infinities, this seems like a possible issue...
          use of exp does introduce a nice non-linearity, 
		but do we really need it?
                
          

  2) can we report the sum-squared-error to see if it improves?

June 6, 2016
Decided to google for help.  Found some math on stackoverflow, but a nice
case-study on classification at 
http://cs231n.github.io/neural-networks-case-study/
which has python numpy code for a softmax one-of-n classifier, and 
does a better job of explaining the loss function, and actually talks through
the entropy part of the calculation.  The example does a batch training,
not stochastic, but I think I can make the conversion.
[I don't think I can do a batch training with 144000 datapoints, anyway -- ]

Also, my regularization loss function doesn't include a factor of 0.5, so it
over-reports the regularization problem; no way the derivative can catch the 
over-reported problem, since its off by a factor of two.  However, making that
change doesn't affect the regularization behavior, I don't know why.
With R=10, L went down for most of the first epoch, then began to climb.

On the loss function front, the github example computes L_i, the Loss function 
for example i, as -log(exp(y[words[3])/sum(exp(y)))
then it defines the batch loss function L as sum(L)/size(L) -- the average of 
the L_i's.

While discussing the code with Swathi Gogula, discovered a couple of bugs,
both seem serious.  
  1) In normalization code in nnEval, divided by sumsq(pt) instead of sum(pt)
  2) in update code for nC array in trainstep, normalized nc instead of nC,
     and simultaneously discarded updates.

Started a run after fixing these two bugs, using R=10 as regularization.
L started low, at 4.46, and declined during the first epoch, with a low of 3.804
near the end of epoch 2, after some milling around, surpassed that with a new
low of 3.7814

It seems to me that 1) we still have the identical weights problem.
                    2) the C array feature weights are very near zero.
                    3) the training is running noticeably slower than hitherto.
                       this may be due to memory usage.  I have a couple of pdf 
		       files open, fair number of browser windows.
		       started epoch 3 about 9:32; looks like about 2 min. per
		       10000 examples.  This is significantly slower than 
		       previously, which was about 40 min for 20 epochs.
Run took until 4:39AM, total > 7 hours.  Mysterious!  What did I do to increase
it so much? Is there some background activity going on?

June 8, 2016
>Firefox uses a significant amount of memory. During that run, firefox RSS was 
31% of physical memory; and since firefox is listening to the net, it apparently
keeps using it.
When I set R=0 and shutdown firefox, as well as extra terminal windows, the
run took about 40 minutes again.  I'll try next with R=1.
The R=0 run didn't come in with L much below 4, but there was laudable variety 
in the C weights; it seemed like maybe some training could be going on, 
although they were all near one another, and about 0.207
The following line shows memory usage (columns 7,8,9)
ps -e v |sed -e 's/^ *//' -e 's/  */  /g' |sort -n --key=8

run with R=1 started at 12:13 and at 12:58 is still in epoch9, this might be
half as fast as R=0,  but I am competing for resources, runnning firefox, edits,
etc. at the same time.  Certainly not using much CPU, but maybe memory

The R=1 configuration is running rapidly toward zero, about a factor of 1e-38 
per epoch.  Edited test.m to allow changing R, stepsize, epochs from commandline
Intention is 
octave test.m R 0.5 stepsize 0.01 epochs 10
but I have  een waiting for the R=1 run to finish before testing it.

Aborted the R=1 run, tried R=0.1, which seems to march toward zero coefficients
at a ratio of 1E-6/epoch.

Here are some thoughts on "lazy" solutions.  
1) if the words were uniformly distributed, then one lazy distribution would
ignore the inputs and set all y outputs equal.  For 100 outputs this would 
give a probability for each output of 0.01, and the correct answer would always
have a probability of 0.01, so that -log(p(w(3))) would be 4.6
2) if the words (as is likely) have a Zipfian distribution, then the second
most common word would occur half as often as the most common one, and the
n-th most common word 1/n as often as the most common.  For this distribution,
ignoring the input, we can minimize L by assigning probabilities
(1/1)/H100, (1/2)/H100,  ... {1/100)/H100 to words in order of frequency, 
[where H100 is the sum of reciprocals from 1 to 100, or 'the 100th harmonic 
number.]
With these probabilities assigned, the expected value of f(w)
is \sum_1^100 f(w)P(w), meaning the expected value of p(w) is \sum p(w)P(w) or
\sum_{i=1}^100 ((1/n)/H100)^2 n \approx 0.0608 and the expected value of L
is 2.8

We could easily prime the bias weights so that the initial state values of
the y units are the log frequencies of the corresponding words.  This might 
be a local minimum, meaning that we could get stuck there and never improve,
but it seems like a plausible strategy.

meantime, it makes sense to randomize the order of presentation of training 
items.  I seem to observe that the value of L seems to fluctuate at about 
the same point in the file on every epoch.


June 8, 2016
Built debug1.m and deb2.m to examine partial results, build confidence in
code.
Carefully paced through steps a, b in Bengio algorithm -- they have some
messed-up rows and columns, but my code seems sound.
Stuck in an extra set of probably unnecessary parentheses in pdiffLwrtO computation, and found a whomping error in C update code.

Running after those corrections doesn't give a dramatic improvement,  but the 
C coefficients finally aren't rushing to identity.  Unfortunately, my run is 
with R=1, and they are rushing toward zero.  Since the paper reports success
with R=1E-4, I'm going to make that the default.

Did a default run after changing the regularization default, and the 
neural net seems to be finally going somewhere.  Took an hour, but I continued
to use the machine, including firefox, while Loss function moved to 2.9.
It's not stably there, but that's after 20 epochs; more might well work better.
I'll try for 150 tonight.
