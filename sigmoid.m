% FUNCTION: sigmoid
% INPUT:    x, matrix
% OUTPUT:   σ(Χ)=1/(1+exp(-x))

function y = sigmoid(x)
y = 1./(1+exp(-x));
end