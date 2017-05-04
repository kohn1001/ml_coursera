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

% Setup some useful variables
%fprintf('x is: %f\n', X);
%fprintf('y is: %f\n', y);
m = size(X, 1);
my = size(y,1) ;
yi = eye(num_labels);
y = yi(y,:);
x = [ones(m, 1) X];
a1 = x;
%size(x)
z2 = x*Theta1';
a2 = sigmoid(z2);
a2b = [ones(size(a2),1) a2];
%size(a2b)
z3 = a2b*Theta2';
a3 = sigmoid(z3);


% You need to return the following variables correctly 
Ji = -y.*log(a3) - (1-y).*log(1-a3);
T1t = Theta1(:,2:end);
T2t = Theta2(:,2:end);
J=((1/m).*sum(sum(Ji)))+(lambda/(2*m)).*(sum(sum(T1t.^2))+sum(sum(T2t.^2)));



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
	%dd2 = zeros(m,1);
%for i=1:m
%	d3 = a3 - y;
%	dd2 = dd2 + d3'*a2;

%forward
%	na1 = [1; X(i,:)'];


%    z2 = Theta1*na1;
%	na2 = [1; sigmoid(z2)];

%j	z3 = Theta2*na2;
%	a3 = sigmoid(z3);
%backward	
%	zgg2 = sigmoidGradient(z2);
%	yi = ([1:num_labels] == y(i))';

%	d3 = a3 - yi;
%	zg2 = [1; sigmoidGradient(z2)];
%	d2 = (Theta2'*d3).*zg2;
%	d2 = d2(2:end);

%	Theta2_grad = Theta2_grad + d3*na2';
%	Theta1_grad = Theta1_grad + d2*na1';

%	d2 = d3*(a2b)';
%end
% Part 3: Implement regularization with the cost function and gradients.

%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


d3 = a3 - y;
d2 = (d3*Theta2 .* sigmoidGradient([ones(size(z2, 1), 1) z2]))(:, 2:end);
%d2 = d2(:, 2:end);


dd1 = d2'*a1;
dd2 = d3'*a2b;


Theta1_grad = dd1./m + (lambda/m)*[zeros(size(Theta1,1), 1) Theta1(:, 2:end)];
Theta2_grad = dd2./m + (lambda/m)*[zeros(size(Theta2,1), 1) Theta2(:, 2:end)];

















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
