% Model wrapper function that implements the training loop. It takes as 
% input the dataset(X), the groundtruth outputs(Y) and all the 
% hyperparameters of the model and returns the trained model 

function parameters = model(X, Y, n_x, n_h, n_y, n_iterations, learning_rate)
    % Initialize parameters
    parameters = initialize_parameters(n_x, n_h, n_y);
    
    % Loop through the number of training iterations
    for iter = 1:n_iterations
        % Perform forward propagation
        [A1, A2] = forward_prop(X, parameters);
        
        % Calculate the cost
        cost = calculate_cost(A2, Y);
        
        % Calculate the gradients
        grads = backward_prop(X, Y, A1, A2, parameters);
        
        % Update parameters with gradient descent
        parameters = update_parameters(parameters, grads, learning_rate);
        
        if mod(iter, 100) == 0
            fprintf('Cost after iteration %d is %f\n', iter, cost)
        end
    end
end