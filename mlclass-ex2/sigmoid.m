function g = sigmoid(z)
%   SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

e_neg_z = exp(-z);
g = 1 ./ (1 + e_neg_z);


% =============================================================

end
