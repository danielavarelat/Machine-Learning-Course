function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
%h1=theta(1).*X(:,1)+theta(2).*X(:,2);
h=X*theta;
J1=(1/(2*m))*sum((h-y).^2);
R=(lambda/(2*m))*sum(theta(2:end).^2);
J=J1+R;

%grad(=(1/m)*sum((h-y).*X(:,1));
%grad(2)=((1/m)*sum((h-y).*X(:,2)))+((lambda/m)*theta(2));
theta0 = [0; theta(2:end)];
grad = (1/m)*(X'*(h-y))+(lambda/m)*theta0;
grad = grad(:);

end
