function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    ntheta = zeros(2, 1);
    for j = 1:2
        for i = 1:m
            K = (theta(1) * X(i, 1)) + (theta(2) * X(i,2)) - y(i);
            K = K * X(i, j);
            ntheta(j) = ntheta(j) + K;
        end
        ntheta(j) = theta(j) - ((alpha * ntheta(j)) / m);
    end
    theta = ntheta;
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
