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


%adding bias unit to input layer
X = [ones(rows(X),1) X];

h2=sigmoid(X*Theta1');

%adding bias unit to hidden layer
h2 = [ones(rows(h2),1) h2];

h3=sigmoid(Theta2*h2');

exy = zeros(num_labels,m);
for i=1:m
   exy( y(i) ,i)=1;
end

J=(1/m)*sum( sum(-exy.*log(h3)-(1-exy).*log(1-h3)) );

stheta1= sum( sum( Theta1(:,2:end).^2 ) );
stheta2= sum( sum( Theta2(:,2:end).^2 ) );

J=J+(lambda/(2*m))*(stheta1+stheta2);

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


delta2=zeros(num_labels, hidden_layer_size+1);
delta1=zeros(hidden_layer_size, input_layer_size+1);

for t=1:m
   a1=X(t,:)';
     z2=Theta1*a1;
   a2=sigmoid(z2);
   a2=[1;a2];
     z3=Theta2*a2;
   a3=sigmoid(z3);

   er3=a3-exy(:,t);
   er2=Theta2'*er3.*sigmoidGradient([1;z2]);
   er2=er2(2:end);

   delta2=delta2+er3*a2';  
   delta1=delta1+er2*a1';    
end

Theta1_grad=(1/m)*delta1;
Theta2_grad=(1/m)*delta2;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


Theta1_grad(:,2:end)=Theta1_grad(:,2:end)+(lambda/m)*Theta1(:,2:end);
Theta2_grad(:,2:end)=Theta2_grad(:,2:end)+(lambda/m)*Theta2(:,2:end);


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
