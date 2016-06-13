% this version attempts to use the notation of Bengio, et al., 
% A Neural probabilistic language model.
% Journal of machine Learning research 3, (2003) pp1137-1155

hidden_layer_size=60
vocabulary_size=2645
feature_length=100
input_order = 4;
R = 1E-4 ;%regularization factor
epochs = 20 ;% number of passes to make over the input
stepsize = 0.001;
output_size = 100;
start_file = "";

% notice that argList is going to be a "cell-array" -- its
% a ragged array of chars.  We can reference individual strings by their row, as
% argList{i}
argList = argv();
skip = 0;
for i=1:nargin
 if (skip == 0)
  if (isequal(argList{i} , "R"))
       R = str2double(argList(i+1));
       skip = 1;
  elseif (isequal(argList{i} , "epochs"))
       epochs = str2double(argList(i+1));
       skip = 1;
  elseif (isequal(argList{i} , "stepsize"))
       stepsize = str2double(argList(i+1));
       skip = 1;
  elseif (isequal(argList{i} , "output_size"))
       output_size = str2double(argList(i+1));
       skip = 1;
  elseif (isequal(argList{i} , "start_file"))
       start_file = str2double(argList(i+1));
       skip = 1;
  else
       printf("unknown parameter: %s\n R and epochs and stepsize and output_size (each followed by a number) are allowed\n",argList{i});
  endif;
 else skip = 0;
 endif
endfor

%display the values which may have been modified by the command arguments

start_time = ctime(time())


epsilon_init = 0.12; % used to initialize weights

if start_file != ""
load -text NN.mat parameters d H b U C
 hidden_layer_size=parameters(1)
vocabulary_size=parameters(2)
feature_length=parameters(3)
input_order =parameters(4)
output_size=parameters(5)
R =parameters(6)

else % startfile name is empty so
%initialize the feature vector to random numbers.
C = rand(vocabulary_size,feature_length)*2*epsilon_init - epsilon_init;
x = zeros(input_order*feature_length,1);

% b and d are biases, and I think breaking them out is unbeautiful.  
% previously I included them in theta1 and theta2 which are now H and U
% I think that the reason for breaking them out is that they
% do not participate in regularization.
d = rand(hidden_layer_size,1)*2*epsilon_init - epsilon_init; %I think it should be a vector...
b = rand(output_size,1)*2*epsilon_init - epsilon_init;

% these aren't really declarations, they're comments about the sizes I expect
output = zeros(output_size, 1);
o = rand(hidden_layer_size);
a = rand(hidden_layer_size);

% weight coefficients
st1 = [size(a,1) size(x,1)]; % second element was 1+size for bias, now in d
H = rand(st1)*2*epsilon_init - epsilon_init; %+(ones(st1)*(-0.5)); % from x to o

st2 = [size(output,1) size(a,1)]; % second element was 1+size for bias, now in b
U = rand(st2)*2*epsilon_init - epsilon_init; %+(ones(st2)*(-0.5)); % from a to output
endif  % of start_file if
% print stuff
R
epochs
stepsize
output_size

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
for reps = 1:epochs
epochSum = 0;
epoch2Sum = 0;
epochCount = 0;

rchoice = randperm(training_rows-4) + 3;
for q = 1:(training_rows-4) % 3:(training_rows-3)
  % Although I make 20 passes over the data, I always do it in a different order
t = rchoice(q);
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

[J, C, d,H, b,U] = trainstep(x,d,H, b,U, C, w, expected, stepsize, R);

epochSum = epochSum + J;
epoch2Sum = epoch2Sum + J*J;
epochCount = epochCount + 1;
runningAverage = runningAverage * 0.99 + J;
if ( mod(cases, 10000) == 0 )
   disp(reps);
   disp(cases);
   disp(runningAverage/100);
%   disp(output);


   
endif
 
endfor
cases % display number of training cases

average = epochSum / epochCount;
variance = (epoch2Sum )/epochCount- average*average;
printf("epoch mean: %f variance %f\n",average, variance);

C_col1_stats = statistics(C(1:end,1)')
C_row1_stats = statistics(C(1:1,1:end))
endfor


%disp(C(1:10,1:end));


%should save the entire learned weights, d, H, b, U, C, as well as 
% the dimensions-determining parameters: 

parameters = [ hidden_layer_size, vocabulary_size, feature_length, input_order , output_size, R , epochs , stepsize ]

save -text NN.mat parameters d H b U C

printf(" started at %s time now %s\n", start_time, ctime(time()));

