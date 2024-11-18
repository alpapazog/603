% Single-sample Neural Network Training with Backpropagation

% Main Function
function NeuralNetForSinglePoint()
    % Single Data Point
    x = [2; 1]; % Input features (column vector)
    y = 3;      % Target output (scalar)

    % Training Parameters
    iterations = 50;         % Number of iterations
    learning_rate = 0.05;    % Learning rate

    % Initialize weights randomly from U(0,1)
    theta1 = rand(2, 2);     % Weights for Layer 1 (2x2)
    theta2 = rand(2, 1);     % Weights for Layer 2 (2x1)

    % Store Loss for Each Iteration
    losses = zeros(iterations, 1);

    % Training Loop
    for iter = 1:iterations
        % Forward Propagation
        [z1, a1, z2, y_hat] = forward_propagation(x', theta1, theta2);

        % Compute Squared Error Loss
        loss = 0.5 * (y - y_hat)^2; % Mean squared error for single sample
        losses(iter) = loss;

        % Backpropagation and Weight Update
        [theta1, theta2] = backpropagation(x', y, z1, a1, z2, y_hat, theta1, theta2, learning_rate);

        % Display Loss
        fprintf('Iteration %d: Loss = %.4f\n', iter, loss);
    end

    % Plot the Loss Curve
    figure;
    plot(1:iterations, losses, 'LineWidth', 2);
    xlabel('Iterations');
    ylabel('Squared Error Loss');
    title('Loss Curve');
    grid on;
end

% Forward Propagation
function [z1, a1, z2, y_hat] = forward_propagation(x, theta1, theta2)
    z1 = x * theta1;         % Linear combination for layer 1
    a1 = sigmoid(z1);        % Activation for layer 1
    z2 = a1 * theta2;        % Linear combination for layer 2
    y_hat = z2;              % Predicted value (output)
end

% Backpropagation
function [theta1, theta2] = backpropagation(x, y, z1, a1, z2, y_hat, theta1, theta2, learning_rate)
    % Gradients for the output layer
    dz2 = y_hat - y;                     % dL/dz2
    dtheta2 = a1' * dz2;                 % dL/dtheta2

    % Gradients for the hidden layer
    dz1 = (dz2 * theta2') .* sigmoid_derivative(z1); % dL/dz1
    dtheta1 = x' * dz1;                    % dL/dtheta1

    % Update parameters
    theta1 = theta1 - learning_rate * dtheta1;
    theta2 = theta2 - learning_rate * dtheta2;
end

% Sigmoid Activation Function
function output = sigmoid(z)
    output = 1 ./ (1 + exp(-z)); % Sigmoid function
end

% Derivative of Sigmoid Function
function output = sigmoid_derivative(z)
    s = sigmoid(z);
    output = s .* (1 - s); % Derivative of sigmoid
end
