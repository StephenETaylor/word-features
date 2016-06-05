% this is a stochastic training function; returns new values of C, d,H, b,U
% which give a higher objective function value than the current ones

function [L,nC, nd,nH, nb,nU] = trainstep ( x, d,H, b,U, C, words, expected, epsilon, R)
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

%compute the Objective function.  Since we're going to be maximizing
% this objective function, I subtract R, so that smaller weights give
% a larger objective function.

% it appears we don't use L except to report it, but
% L = (1/T)*Sum log f(w_t, w_{t-1},...,w_{t-n}; Theta) + R(Theta)
% I think T is the (constant) size of the vocabulary
% I omit that calculation, and I really don't understand the sum,
% although I think the intent is to compute the per-word entropy
L= sum(y) - R*sum(sum(abs(U))) -R*sum(sum(abs(H))) - R*sum(sum(abs(C)));

%compute the gradients
% paper shows loop over each of the outputs p[j] or the corresponding y[j]
% and my comments reflect that, but the code attempts to combine the loop
% into matrix operations.

% partial derivative of L wrt y[j] = (j==words[3]?1:0) - p[j]
pdiffLwrtY = expected - p; % 

% new b[j]  = b[j] + epsilon * pdiffLwrtY[j]
nb = b + epsilon*pdiffLwrtY;   % adjust bias for next 

%I chose not to implement direct connections: no W[j] weights to change
% and pdiffLwrtX not immediately changed by pdiffLwrtY

% pdiffLwrtA += pdiffLwrtY * U[j]  % U[j] is a vector, better be hidden_layer_size
pdiffLwrtA = U' * pdiffLwrtY;  %sum over all y with matrix multiply

% U[j] = U[j] + epsilon * pdiffLwrtY[j] * a

nU = U + epsilon * (pdiffLwrtY * a');

%  the (b) comment (b)calls for summing and sharing pdiffLwrtX and pdiffLwrtA
%  across all processors, but although I'm following along, I'm not writing
%  networked code.

%  (c) back-prop step.  For all k:
%  pdiffLwrtO[k] = (1-a[k]**2) * pdiffLwrtA[k]

pdiffLwrtO = (ones(size(a)) - a .* a) .* pdiffLwrtA;

% paper has:  
% pdiffLwrtX = pdiffLwrtX + H' * pdifLwrtO
% but they initialized to zero, then share and sum pdiffLwrtX among processors
% as well as possibly having direct connects from x to y.
% So I just initialize pdiffLwrtX here

pdiffLwrtX =  H' * pdiffLwrtO;

nd = d + epsilon * pdiffLwrtO;

nH = H + epsilon * pdiffLwrtO * x';

nC = C;
nC(words(1),1:100) = C(1) + epsilon*pdiffLwrtX(1:100);
nC(words(2),1:100) = C(2) + epsilon*pdiffLwrtX(101:200);
nC(words(4),1:100) = C(4) + epsilon*pdiffLwrtX(201:300);
nC(words(5),1:100) = C(5) + epsilon*pdiffLwrtX(301:400);

%return the corrected values
%return
endfunction
