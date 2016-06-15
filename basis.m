% compute a basis for the word weights
% I anticipate that the eigenvectors of the variance matrix will be such a 
% basis, although perhaps not a useful one.

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

n = size(C,1); %number of different words
M = mean(C);  % get a vector of column means
D = C-M; % I should maybe explicitly cast M to correct dimensions?
S = (1/n)*(D' * D); % result is a feature_length by feature_length matrix

% now get eigenvectors
[E, Lambdas] = eigs(S,90); %get 90 most interesting eigenvectors.
for i=1:90
   Lambdas(i,i)
endfor

keyboard()

