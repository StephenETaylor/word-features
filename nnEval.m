function [s,y,p, o, a ]  = nnEval(x, d,H, b,U);
% x is  px1 column vector
% d is q x 1  bias vector
% H is a q x p matrix
% there's a hidden layer, q x 1
% for which we compute o and p

o = H * x + d;
a = tanh(o);

%U is a r x q  matrix
%output is a r x 1 column vector
y = b + (U * a);
pt = exp(y);
s = sumsq(pt);  % here we approximate s by summing only over first 1000 words
p = (1/s) * pt;

%return
endfunction

