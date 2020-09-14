% FUNCTION: predict
% INPUTS:   X: Feature matrix of size (n_x, n_examples)
%           parameters: trained model parameters
% OUTPUTS:  y_pred: predictions of size (n_y, n_examples)

function y_pred = predict(X, parameters)
    % Forward pass the input
    [~, A2] = forward_prop(X, parameters);
    
    % All predictions that have activation value grater that 0.5 are
    % consindered as positive label(1), else negative label(0)
    y_pred = A2;
    y_pred(A2 > 0.5) = 1;
    y_pred(A2 <= 0.5) = 0;
end