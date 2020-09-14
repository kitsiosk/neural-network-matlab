% FUNCTION: backward_prop
% INPUTS:
%           X: Input vector of size (n_x, n_examples)
%           Y: Groundtruth vector of size(n_y, n_examples)
%           A1: Activations of the first layer of size (n_h, n_examples)
%           A2: Activations of the second layer of size (n_y, n_examples)
%           parameters: Network parameters. W2 will be used for gradient
%                       calculation
% OUTPUTS:
%           grads:  Gradient of the parameters calculated using 
%                   backpropagation algorithm
% This function performs the backpropation algorithm, which is actualy the
% chain rule for multivariate input/output functions.

function grads = backward_prop(X, Y, A1, A2, parameters)
    % Number of training examples
    m = length(Y);

    W2 = parameters.W2;
    
    dZ2 = A2 - Y;
    dW2 = (dZ2 * A1')/m;
    db2 = sum(dZ2, 2)/m;
    dZ1 = (W2' * dZ2) .* (1 - A1.^2);
    dW1 = (dZ1 * X')/m;
    db1 = sum(dZ1, 2)/m;
    
    grads = {};
    grads.dW1 = dW1;
    grads.db1 = db1;
    grads.dW2 = dW2;
    grads.db2 = db2;
end