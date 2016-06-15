% debugging file

%read in saved parameters
load -text NN.mat parameters d H b U C
 hidden_layer_size=parameters(1)
vocabulary_size=parameters(2)
feature_length=parameters(3)
input_order =parameters(4)
output_size=parameters(5)
R =parameters(6)
epochs =parameters(7)
stepsize =parameters(8)

start_time = ctime(time())



% these aren't really declarations, they're comments about the sizes I expect
%output = zeros(feature_length, 1);
%o = rand(hidden_layer_size);
%a = rand(hidden_layer_size);



%run through the training file.  open to read text using default architecture
%[FID, MSG] = fopen("train.txt", "rt")
emma = textread("train.txt", "%f");

rows = size(emma,1);
%display (rows);

%we'll train with only 80 percent of the data
% I haven't written code for validation and test, but I'm reserving 10% for each 
training_rows = floor(0.8 * rows);

for i = 1:(rows)
   if (emma(i) == 0 )
      emma(i) = vocabulary_size; 
   endif
endfor


%we may repeat this many times; but for starters, a single pass
w = zeros(5,1);
cases = 0;
runningAverage = 0;
for t =  3:(training_rows-3)
% input order is hardcoded in this loop (as 4 words of context)
    
for i = -2:2
w(i+3) = emma(t+i);
endfor

if (w(3) > output_size )  % changing action ...
  continue; 
endif
cases = cases + 1;

C1 =  C(w(1),1:feature_length);
C2 =  C(w(2),1:feature_length);
C4 =  C(w(4),1:feature_length);
C5 =  C(w(5),1:feature_length);
x = [ C1 C2 C4 C5 ]';
expected =  zeros(output_size,1);
expected(w(3)) = 1;

%[J, C, d,H, b,U] = trainstep(x,d,H, b,U, C, w, expected, stepsize, R);
% this is a stochastic training function; returns new values of C, d,H, b,U
% which give a higher objective function value than the current ones

words = w;
epsilon = R;
%function [L,nC, nd,nH, nb,nU] = trainstep ( x, d,H, b,U, C, words, expected, epsilon, R)
%x is input units,  built up in caller from C and words
%d is bias for hidden layer
%H is weight coefficients to build hidden layer o
%a is hidden units, a = tanh(o)
%b is bias for y layer
%U is weight coefficients to build y layer
%y = b+U*a, and said to be log probabilities units.
%p is normalized e**y/s, 



%evaluate the neural network. 
[s,y,p, o,a] = nnEval(x, d,H, b,U);

%compute the Objective function.  
% The paper says we're going to be maximizing
% this objective function: 
% L = (1/T)*Sum log f(w_t, w_{t-1},...,w_{t-n}; Theta) + R(Theta)
%  ( T is the (constant) size of the vocabulary)

% which is the negative of the per-word entropy of the entire corpus, plus a
% regularization function.
% In order to approximate my understanding of what needs to happen,
% I replace y_t = f(w_t,...) with p[word[3]], the probability computed for 
% the correct answer, (that is, the appropriately normalized exp(y))
% AND I omit the division by T, since instead of considering the entire
% batch of (144000) examples, we're only evaluating a single example (stochastic% training).  For the regularization penalty, R(Theta) in the objective
% function, I use a sum of squares of the weight coefficients times a 
% regularization constant R/2; in the backprop code I decay the coeffs by
% (1-R*epsilon) as hinted in the paper.
% instead of letting the function L run from -inf to 0, I negate the
% value so it runs from 0 (when correct answer chosen) to +inf (when 
% normalized probability of the correct answer is 0)

% it appears we don't use L except to report it, but
% but the signs of the partialDiffLwrt* might seem to actually be maximizing
% (hill-climbing) instead of descending, since the code has, e.g.: 
%   nU = (1-R*epsilon)*(U + epsilon * (pdiffLwrtY * a'));
% instead of ... - epsilon...

L= -(log(p(words(3)))) + 0.5*R*(sum(sumsq(U)) +sum(sumsq(H)) + sum(sumsq(C)));


%compute the gradients
% paper shows loop over each of the outputs p[j] or the corresponding y[j]
% and my comments reflect that, but the code attempts to combine the loop
% into matrix operations.

% partial derivative of L wrt y[j] = (j==words[3]?1:0) - p[j]
pdiffLwrtY = expected - p; %sic. Bishop, Ng reverse sign, use ...-epsilon...


% new b[j]  = b[j] + epsilon * pdiffLwrtY[j]
nb = b + epsilon*pdiffLwrtY;   % adjust bias for next 

%I chose not to implement direct connections: no W[j] weights to change
% and pdiffLwrtX not immediately changed by pdiffLwrtY

% pdiffLwrtA += pdiffLwrtY * U[j]  % U[j] is a vector, better be hidden_layer_size
%  ptLa attempts to duplicate Bengio calculation
%  finally got it to match: note transpose in my loop, not in paper.
ptLa = zeros(hidden_layer_size,1);
for j = 1:output_size
  ptLa = ptLa + (pdiffLwrtY(j)) * (U(j,1:hidden_layer_size))';
endfor

% here I replace loop
pdiffLwrtA = U' * pdiffLwrtY;  %sum over all y with matrix multiply

% U[j] = U[j] + epsilon * pdiffLwrtY[j] * a

% regularization.  Should the learning adjustment come before or after decay?
%   I put it before, since y was computed with the old U.
%   if R >1, regularization swamps learning otherwise
%   Bengio used R = 1e-4 and 1e-5, 
%               epsilon = 0.001, epsilon decay of epsilon/(1+1e8)
%   I don't currently have any epsilon decay code.
nU = (1-R*epsilon)*(U + epsilon * (pdiffLwrtY * a'));

%  the (b) comment (b)calls for summing and sharing pdiffLwrtX and pdiffLwrtA
%  across all processors, but although I'm following along, I'm not writing
%  networked code.

%  (c) back-prop step.  For all k:
%  pdiffLwrtO[k] = (1-a[k]**2) * pdiffLwrtA[k]

pdiffLwrtO = (ones(size(a)) - (a .* a)) .* pdiffLwrtA;

% paper has:  
% pdiffLwrtX = pdiffLwrtX + H' * pdifLwrtO
% but they initialized to zero, then share and sum pdiffLwrtX among processors
% as well as possibly having direct connects from x to y.
% So I just initialize pdiffLwrtX here

pdiffLwrtX =  H' * pdiffLwrtO;

nd = d + epsilon * pdiffLwrtO;

%regularization:
nH = (1-R*epsilon)*(H + epsilon * pdiffLwrtO * x');


nC = C;
nC(words(1),1:100) = C(words(1),1:end) + epsilon*pdiffLwrtX(1:100);
nC(words(2),1:100) = C(words(2),1:end) + epsilon*pdiffLwrtX(101:200);
nC(words(4),1:100) = C(words(4),1:end) + epsilon*pdiffLwrtX(201:300);
nC(words(5),1:100) = C(words(5),1:end) + epsilon*pdiffLwrtX(301:400);

%regularization:
nC = nC*(1-R*epsilon);

keyboard()
%return the corrected values
%return
% if we were training we'd save nb nH nd nU nC in their corresponding spots.
endfor


%disp(C(1:10,1:end));


%should save the entire learned weights, d, H, b, U, C, as well as 
% the dimensions-determining parameters: 

%parameters = [ hidden_layer_size, vocabulary_size, feature_length, input_order , output_size, R , epochs , stepsize ]

%save -text NN.mat parameters d H b U C

printf(" started at %s time now %s\n", start_time, ctime(time()));

