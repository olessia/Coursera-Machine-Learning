function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);



% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% add the bias term
X = [ones(m, 1) X];

input_into_h1 = X*Theta1';
state_h1 = sigmoid(input_into_h1);

% add the bias term
state_h1 = [ones(size(state_h1), 1) state_h1];

input_into_output = state_h1 * Theta2';
state_output = sigmoid(input_into_output);

% find the index of the largest value in each row
[value, index] = max(state_output, [], 2);
p = index;



% =========================================================================


end
