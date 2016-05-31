function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%

%    some setup
%

%    num_labels is ymax = max(y);

yLogical = zeros (size(y,1), num_labels);
yLogical = (0 == 1);
for i = 1:(size(y,1))
    yLogical(i,y(i)) = (1==1);
end 


% add the column of 1's to X for the bias coefficients
X = [ones(m,1) X];

% compute the A^2 array; result should be i x hidden_layer_size
z1 =   X * Theta1';
hTheta1 = [ones(m,1) sigmoid( z1)];



% compute the output array A^3; result should by i x num_labels
z2 = (hTheta1 * Theta2');
hTheta2 = sigmoid(z2);


unsummed = -yLogical .* log (hTheta2) - (1-yLogical) .* log(1-hTheta2); % result is m x K
summedK =  unsummed * ones(num_labels,1); % result is m x 1

summedM = ones(1,m) * summedK ; % result is 1 x 1

J = 1/m * summedM; % a scalar

% regularization term:  (\lambda/2/m)(\sum^{hidden_layer_size}_j
%                                      \sum^{input_layer_size}_k
%                                         Theta1(j,k)^2
% + \sum^{output_layer_size}^j \sum^{hidden_layer_size}_k Theta2(j,k)^2
%
%  size(Theta1) = {hidden_layer_size} x {input_layer_size+1}

% make copy of Thetas, omitting the bias coefficients
T1 = Theta1(1:hidden_layer_size, 2:input_layer_size+1);
T2 = Theta2(1:num_labels, 2:hidden_layer_size+1);

% computation includes whole array:
regterm = (lambda/2/m)*(sum(sumsq(T1)) + sum(sumsq(T2)));
J = J + regterm;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

% next compute gradients.

littleDelta3 = hTheta2 - yLogical ;% result is m x num_labels
% hTheta2(1:2, :) %debug
% yLogical(1:2, :) %debug
% littleDelta3(1:2, :) %debug
%   szlD3 = size(littleDelta3)
%   szhT2 = size(hTheta2)


% note that hTheta1 is a2, the hiddenlayer
ld21 = (littleDelta3 * Theta2);
ld22 =   hTheta1(1:m,1:hidden_layer_size+1) ;
ld23 =    (ones(m,hidden_layer_size+1)-hTheta1(1:m,1:hidden_layer_size+1));
littleDelta2 = ld21(1:m,1:hidden_layer_size+1) .* ld22 .* ld23;
% littleDelta2(1:2,:) %debug
for i = 1 : m	%size(X,1)
   % compute Delta^l = Delta^l + littleDelta^{l+1} * a^l'
   littleJoe = (littleDelta3(i,1:num_labels)' * hTheta1(i,:));
%   if (i == 1 || i == 2 ) 
%      disp('littleJoe='),disp(littleJoe) 
%   endif %debug
   Theta2_grad = Theta2_grad .+ littleJoe;
   Theta1_grad = Theta1_grad .+ littleDelta2(i, 2:hidden_layer_size+1)' * X(i,:);
   
end % end for i
Theta1_grad = 1/m * Theta1_grad;
Theta2_grad = 1/m * Theta2_grad;



% % Part 3: Implement regularization with the cost function and gradients.  
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.


% -------------------------------------------------------------
% so add in the regularization:

Theta1_grad = Theta1_grad + (lambda/m) * [zeros(size(Theta1,1),1) Theta1(:,2:end)]
Theta2_grad = Theta2_grad + (lambda/m) * [zeros(size(Theta2,1),1) Theta2(:,2:end)]


% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
