function [output, ah, hid1, inp1]  = nnEval(input, theta1, theta2);
% input is  px1 column vector
inp1 = [1; input] ; % add bias
% theta1 is a q x p+1 matrix
% there's a hidden layer, q x 1
%hidden = sigmoid(theta1 * inp1);

ah = theta1 * inp1;
hidden = tanh(ah);
hid1 = [1; hidden];

%theta2 is a r x q+1  matrix
%output is a r x 1 column vector
output = (theta2 * hid1);

%return
endfunction

