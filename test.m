% this version attempts to use the notation of Bengio, et al., 
% A Neural probabilistic language model.
% Journal of machine Learning research 3, (2003) pp1137-1155

hidden_layer_size=60
vocabulary_size=2645
feature_length=100
input_order = 4;
R = 0; %regularization factor
output_size = 100;
"using one-of-",output_size

epsilon_init = 0.12; % used to initialize weights

%initialize the feature vector to random numbers.
C = rand(vocabulary_size,feature_length)*2*epsilon_init - epsilon_init;
x = zeros(input_order*feature_length,1);

% b and d are biases, and I think breaking them out is unbeautiful.  
% previously I included them in theta1 and theta2 which are now H and U
d = rand(hidden_layer_size,1)*2*epsilon_init - epsilon_init; %I think it should be a vector...
b = rand(output_size,1)*2*epsilon_init - epsilon_init;

% these aren't really declarations, they're comments about the sizes I expect
output = zeros(feature_length, 1);
o = rand(hidden_layer_size);
a = rand(hidden_layer_size);

% weight coefficients
st1 = [size(a,1) size(x,1)]; % second element was 1+size for bias, now in d
H = rand(st1)*2*epsilon_init - epsilon_init; %+(ones(st1)*(-0.5)); % from x to o

st2 = [size(output,1) size(a,1)]; % second element was 1+size for bias, now in b
U = rand(st2)*2*epsilon_init - epsilon_init; %+(ones(st2)*(-0.5)); % from a to output

%run through the training file.  open to read text using default architecture
%[FID, MSG] = fopen("train.txt", "rt")
emma = textread("train.txt", "%f");

rows = size(emma,1);
%display (rows);

%we'll train with only 80 percent of the data
% I haven't written code for validation and test, but I'm reserving 10% for each 
training_rows = floor(0.8 * rows);

for i = 1:(training_rows)
   if (emma(i) == 0 )
      emma(i) = vocabulary_size;
   endif
endfor
%we may repeat this many times; but for starters, a single pass
w = zeros(5,1);
cases = 0;
runningAverage = 0;
stepsize = 0.001;
for reps = 1:20
for t = 3:(training_rows-3)
  % Although I make 20 passes over the data, I always do it in the same order
  % it would make more sense to incorporate some randomness to chose which
% input order is hardcoded in this loop (as 4 words of context)
    
for i = -2:2
w(i+3) = emma(t+i);
endfor

if (w(3) >= output_size )  % changing action ...
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

[J, C, d,H, b,U] = trainstep(x,d,H, b,U, C, w, expected, stepsize, R);
runningAverage = runningAverage * 0.99 + J;
if ( mod(t, 10000) == 0 )
   disp(reps);
   disp(t);
   disp(runningAverage/100);
%   disp(output);
   
endif
 
endfor
disp(cases);
disp(runningAverage/100);
disp(C(1:2,1:end));
disp(C(99:100,1:end));
endfor


disp(C(1:10,1:end));
save -ascii C.mat C


