function g = sigmoid(z)
%SIGMOID Compute sigmoid functoon
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).


si = size(z);

if si==[1,1],
   g = 1/(1 + exp(-z));
else if si(2)==1,
   for i=1:si(1),
      g(i)=1/(1 + exp(-z(i)));

   end   
else
   
   for i=1:si(1),
      for j=1:si(2),
         g(i,j)=1/(1 + exp(-z(i,j)));

      end
   end   

end

% =============================================================

end

