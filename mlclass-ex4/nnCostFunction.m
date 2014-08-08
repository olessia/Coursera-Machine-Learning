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

% number of training cases
m = size(X, 1);


% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
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
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% -------------------------------------------------------------

identity = eye(num_labels);    
y_vectors = identity(y,:);   

% add the bias term
X = [ones(m, 1) X];

hypothesis = NN_forward_pass(X, Theta1, Theta2);

% unregularized cost
J = sum(sum((-y_vectors .* log(hypothesis) - (1 - y_vectors) .* log(1 - hypothesis))))/m;

% regularization parameter (not including the bias)
Theta1_no_bias = Theta1(:, 2:end);
Theta2_no_bias = Theta2(:, 2:end);
regularize = (sum(sum(Theta1_no_bias .^ 2)) + sum(sum(Theta2_no_bias .^ 2))) * lambda / 2 / m;

% regularized cost
J = J + regularize;

Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% iterate over training cases
% for t = 1:m
% 	a1 = X(t,:);
% 	y_vector = identity(y(t), :);
	
% 	% forward pass
% 	% calculate the state of hidden layer 1
% 	z2 = a1*Theta1';
% 	a2 = sigmoid(z2);

% 	% add the bias term
% 	a2 = [ones(size(a2, 1), 1) a2];

% 	% calculate the state of hidden layer 2
% 	z3 = a2 * Theta2';
% 	a3 = sigmoid(z3);

% 	beta3 = (a3 - (1 == y_vector))';
% 	beta2 = ((Theta2(:, 2:end))' * beta3)' .* sigmoidGradient(z2);

% 	Theta1_grad = Theta1_grad + beta2' * a1;
% 	Theta2_grad = Theta2_grad + beta3 * a2;
% end

% vectorized
% calculate the state of hidden layer 1
z2 = X*Theta1';
a2 = sigmoid(z2);

% add the bias term
a2 = [ones(size(a2, 1), 1) a2];

% calculate the state of hidden layer 2
z3 = a2 * Theta2';
a3 = sigmoid(z3);

beta3 = a3 - (1 == y_vectors);
beta2 = ((Theta2(:, 2:end))' * beta3')' .* sigmoidGradient(z2);

Theta1_grad = beta2' * X / m;
Theta2_grad = beta3' * a2 / m;

% regularize gradients
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + Theta1(:, 2:end) * lambda / m;
Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + Theta2(:, 2:end) * lambda / m;


% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
