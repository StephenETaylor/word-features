hidden_layer_size=60
vocabulary_size=2645
feature_length=100
input_order = 4;

epsilon_init = 0.12; % used to initialize weights

%initialize the feature vector to random numbers.
C = rand(vocabulary_size,feature_length);
input = zeros(input_order*feature_length);
output = zeros(feature_length);
hidden = rand(hidden_layer_size);

% weight coefficients
st1 = [size(hidden,1) 1+size(input,1)];
theta1 = rand(st1)*2*epsilon_init - epsilon_init; %+(ones(st1)*(-0.5)); % from input to hidden

st2 = [size(output,1) 1+size(hidden,1)];
theta2 = rand(st2)*2*epsilon_init - epsilon_init; %+(ones(st2)*(-0.5)); % from hidden to output

%run through the training file.  open to read text using default architecture
%[FID, MSG] = fopen("train.txt", "rt")
emma = textread("train.txt", "%f");

rows = size(emma,1);
%display (rows);

%we'll train with only 80 percent of the data
training_rows = floor(0.8 * rows);

for i = 1:(training_rows)
   if (emma(i) == 0 )
      emma(i) = vocabulary_size;
   endif
endfor
%we may repeat this many times; but for starters, a single pass
w = zeros(5,1);
runningAverage = 0;
stepsize = 0.001;
for i = 1:(training_rows-5)
% input order is hardcoded in this loop (as 4 words of context)
w(1) = emma(i);
w(2) = emma(i+1);
w(3) = emma(i+2);
w(4) = emma(i+3);
w(5) = emma(i+4);

C1 =  C(w(1),1:feature_length);
C2 =  C(w(2),1:feature_length);
C3 =  C(w(3),1:feature_length);
C4 =  C(w(4),1:feature_length);
C5 =  C(w(5),1:feature_length);
input = [ C1 C2 C4 C5 ]';
y =  C3';

[J, C, theta1, theta2] = trainstep(input,theta1, theta2, C, w, y, stepsize);
runningAverage = runningAverage * 0.99 + J;
if ( mod(i, 1000) == 0 )
   disp(runningAverage/100);
endif
 
endfor

disp(C);

