% Neural Network Backpropagation for Regression with Squared Error Loss

% Initialization
function NeuralNetBackprop()
    % Input Data (Example Dataset)
    x_train = [0.1, 0.2; 0.4, 0.6; 0.5, 0.9; 0.8, 0.1]; % Input data (2 features)
    y_train = [0.3; 0.5; 0.7; 0.9];                    % Target values (1 output)

    % Training Parameters
    epochs = 1000;            % Number of epochs
    learning_rate = 0.1;      % Learning rate

    % Train the neural network
    [theta1, theta2, losses] = train_neural_network(x_train, y_train, epochs, learning_rate);

    % Plot the loss curve
    figure;
    plot(1:epochs, losses, 'LineWidth', 2);
    xlabel('Epochs');
    ylabel('Loss');
    title('Training Loss');
    grid on;
end

% Sigmoid activation function and its derivative
function output = sigmoid(z)
    output = 1 ./ (1 + exp(-z)); % Sigmoid function
end

function output = sigmoid_derivative(z)
    s = sigmoid(z);
    output = s .* (1 - s); % Derivative of sigmoid
end

% Initialize parameters
function [theta1, theta2] = initialize_parameters()
    rng(42); % For reproducibility
    theta1 = randn(2, 2) * 0.01; % Weights for Layer 1 (2x2)
    theta2 = randn(2, 1) * 0.01; % Weights for Layer 2 (2x1)
end

% Forward propagation
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

% Train the neural network
function [theta1, theta2, losses] = train_neural_network(x_train, y_train, epochs, learning_rate)
    % Initialize parameters
    [theta1, theta2] = initialize_parameters();
    losses = zeros(epochs, 1);

    for epoch = 1:epochs
        % Forward pass
        [z1, a1, z2, y_hat] = forward_propagation(x_train, theta1, theta2);

        % Compute loss
        loss = 0.5 * mean((y_train - y_hat).^2); % Mean squared error
        losses(epoch) = loss;

        % Backward pass
        [theta1, theta2] = backpropagation(x_train, y_train, z1, a1, z2, y_hat, theta1, theta2, learning_rate);

        % Display loss every 100 epochs
        if mod(epoch, 100) == 0
            fprintf('Epoch %d: Loss = %.4f\n', epoch, loss);
        end
    end
end
