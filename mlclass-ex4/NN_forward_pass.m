function [hypothesis] = NN_forward_pass(X, Theta1, Theta2)

% Calculate the forward pass of the neural network with two hidden layers
% X is a matrix where each row is a training example
% Theta1 and Theta2 are the weights

% calculate the state of hidden layer 1
input_into_h1 = X*Theta1';
state_h1 = sigmoid(input_into_h1);

% add the bias term
state_h1 = [ones(size(state_h1, 1), 1) state_h1];

% calculate the state of hidden layer 2
input_into_output = state_h1 * Theta2';
hypothesis = sigmoid(input_into_output);

end