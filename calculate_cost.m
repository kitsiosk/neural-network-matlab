% FUNCTION: calculate_cost
% INPUTS:   A2: Output of the NN for a given input of size 
% (n_y, n_examples) where n_examples is the number of examples of the
% input(batch size)
%           Y: Groundtruth vector of size (n_y, n_examples)
% OUTPUTS:  Logarithmic cross entropy loss between A2 and Y. 
% Higher loss value means predictions in A2 are not close to groundtruth
% values in Y

function cost = calculate_cost(A2, Y)
    % Number of training examples
    m = length(Y);
    % Add epsilon in log values to avoid numerical errors
    epsilon = 1e-8;
    cost = -sum( Y.*log(A2 + epsilon) + (1-Y).*log(1 - A2 + epsilon) )/m;
end