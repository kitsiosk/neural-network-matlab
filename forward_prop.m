% FUNCTION: forward_prop
%           Propagates the input through the NN to obtain the output.
%           Propagation equations are known from theory. The two
%           activations functions used are sigmoid and tanh.
% INPUTS:   X: Feature vector or matrix(From our dataset or a new example
%           for prediction
% OUTPUTS:  A1: Activation vector of the first layer
%           A2: Activation vector of the second layer. This one will be
%           used for prediction


function [A1, A2] = forward_prop(X, parameters)
    W1 = parameters.W1;
    b1 = parameters.b1;
    W2 = parameters.W2;
    b2 = parameters.b2;
    
    Z1 = W1*X + b1;
    A1 = sigmoid(Z1);
    Z2 = W2*A1 + b2;
    A2 = tanh(Z2);
end