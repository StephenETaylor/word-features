function [J,nC,ntheta1,ntheta2] = trainstep ( input, theta1, theta2, C, words, y, stepsize)

% a stochastic training function; returns new values of C, theta1, theta2
% which are give a lower error function value than the current ones

%evaluate the neural network. hidden and x both have the 1 entries for bias
[output, ah, hidden, x] = nnEval(input, theta1, theta2);

%compute the error function
diffs = output - y;
J= diffs'*diffs;		% sum of squares

% I have omitted regularization, as well as another problem:
%   Any version of C in which all rows are equal has a zero error code!
%   So the error function should penalize low variance in C.
%   The Bengio et al. article avoids this by using one-of-n encoding
%   for the output.

%compute the gradients
deltaY = output - C(words(3));
deltaC3 = -1*ones(size(C,2));

%        theta2Grad[j,i] = partialDiff(J,theta2[j,i])
%			 = deltaY[j] * hidden[i]
theta2Grad = deltaY * hidden';
%
%        deltaH[j] = diff(J, ah[j]) 
%		   = (diff(tanh,x)(ah[j])) * sum(all k, theta2[k,j]*deltaY)
deltaH = (ones(size(hidden)) - (hidden .* hidden)) .* ( theta2'*deltaY ) ;

%        theta1Grad[j,i] = partialDiff(J,theta1[j,i])
%			 = deltaH[j] * input[i]
theta1Grad = deltaH(2:end,1) * x';

%	since we don't do a forward computation of hidden[1] (==1, the 'bias')
%	there aren't corresponding theta1 values to compute it.
%	(not an issue for deltaY: since y is output there is no y bias unit)
deltaX = theta1' * deltaH(2:end,1);

%correct the various coefficients
ntheta2 = theta2 - stepsize* theta2Grad;
ntheta1 = theta1 - stepsize* theta1Grad;
nC = C;
nC(words(1),1:100) = C(1) - stepsize*deltaX(2:101);
nC(words(2),1:100) = C(2) - stepsize*deltaX(102:201);
nC(words(4),1:100) = C(4) - stepsize*deltaX(202:301);
nC(words(5),1:100) = C(5) - stepsize*deltaX(302:401);
nC(words(3),1:100) = C(3) - stepsize*deltaY;

%return the corrected values
%return
endfunction
