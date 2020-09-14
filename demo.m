% Demo program for intuitively building a Neural Network(NN) from scratch.
% We will construct a NN with 3 layers in total: 1 input layer, 1 hidden
% layer and 1 output layer with dimensions n_x, n_h, n_y. 

%For demonstration
% purposes, a rather simple dataset is used i.e. the XOR function dataset
% which is stored in X. Each column of the dataset is a training example
% for our NN and in total we have 4 training examples(all the possible
% binary inputs in the XOR function). Vector Y contains the desired
% result(groundtruth) at each respective column. 
X = [0 0 1 1;
     0 1 0 1];
Y = [0 1 1 0];

% Since our inputs have 2
% features and our outputs 1 element we set n_x=2 and n_y=1. Parameters n_h
% is the number of hidden units in the hidden layer and is a
% hyperparameter hence you can try different values
n_x = 2;
n_h = 2;
n_y = 1;

% The learning rate and number of iterations are also 2 hyperparameters.
% Some good default values are those below
learning_rate = 0.3;
n_iterations = 1000;

% After having the environment set, lets train our model and retrieve the
% trained parameters for future predictions
trained_parameters = model(X, Y, n_x, n_h, n_y, n_iterations, learning_rate);

% Make a prediction for a new example(generally unseen) 
x_new = [0; 1];
y_pred = predict(x_new, trained_parameters);
fprintf('Model prediction for %d XOR %d is %d\n', x_new(1), x_new(2), y_pred)

%
% AUTHOR        Kitsios Konstantinos        kitsiosk@ece.auth.gr
%