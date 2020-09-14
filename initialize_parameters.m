% FUNCTION: initialize_parameters
% INPUT:    Model architecture i.e. number of units in each layer. This
%           function only words for 3 layer NN
% OUTPUT:   Initialized parameters of the NN. Weight matrices are initialized
%           randomly in [0, 1] while bias vectors are initialized to zero

function parameters = initialize_parameters(n_x, n_h, n_y)
 W1 = rand(n_h, n_x);
 b1 = zeros(n_h, 1);
 W2 = rand(n_y, n_h);
 b2 = zeros(n_y, 1);
 
 parameters = {};
 parameters.W1 = W1;
 parameters.b1 = b1;
 parameters.W2 = W2;
 parameters.b2 = b2;
end