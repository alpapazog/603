% Load data
data = readtable('Pizza.csv');

% Extract features (columns 3 to 9)
X = table2array(data(:, 3:9));

% Normalize the data
X = normalize(X);

% Define the number of samples and features
[n, d] = size(X);

% Preallocate for MSE values
mse_linear = zeros(6, 1);

% Loop over code dimensions h
for h = 1:6
    % Encoder: Linear transformation to reduced dimensions
    W_encoder = randn(d, h); % Initialize weights for encoder
    H = X * W_encoder;       % Encoded representation (n x h)
    
    % Decoder: Linear reconstruction
    W_decoder = randn(h, d); % Initialize weights for decoder
    X_hat = H * W_decoder;   % Reconstructed input (n x d)
    
    % Calculate MSE
    mse_linear(h) = (1/n) * norm(X - X_hat, 'fro')^2;
end

% Plot MSE vs h for the linear autoencoder
figure;
plot(1:6, mse_linear, '-o');
title('Linear Autoencoder: MSE vs Code Dimension h');
xlabel('Code Dimension (h)');
ylabel('Mean Squared Error (MSE)');
grid on;

% Define ReLU activation function
relu = @(x) max(0, x);

% Preallocate for MSE values
mse_relu = zeros(6, 1);

% Train ReLU-based autoencoder for varying h
for h = 1:6
    % Initialize weights and biases
    W_encoder = randn(d, h); % Encoder weights
    b_encoder = randn(1, h); % Encoder bias
    W_decoder = randn(h, d); % Decoder weights
    b_decoder = randn(1, d); % Decoder bias
    
    % Training parameters
    learning_rate = 0.01;
    epochs = 1000;
    
    for epoch = 1:epochs
        % Forward pass
        H = relu(X * W_encoder + b_encoder); % Encode with ReLU
        X_hat = H * W_decoder + b_decoder;  % Decode
        
        % Compute gradients (simple backpropagation)
        error = X - X_hat; % Error matrix: n x d
        
        % Decoder gradients
        dW_decoder = -2 * (H' * error) / n; % h x d
        db_decoder = -2 * sum(error, 1) / n; % 1 x d
        
        % Encoder gradients
        delta = (error * W_decoder') .* (H > 0); % n x h
        dW_encoder = -2 * (X' * delta) / n; % d x h
        db_encoder = -2 * sum(delta, 1) / n; % 1 x h
        
        % Update weights and biases
        W_decoder = W_decoder - learning_rate * dW_decoder;
        b_decoder = b_decoder - learning_rate * db_decoder;
        W_encoder = W_encoder - learning_rate * dW_encoder;
        b_encoder = b_encoder - learning_rate * db_encoder;
    end
    
    % Recompute final reconstruction and MSE
    H = relu(X * W_encoder + b_encoder);
    X_hat = H * W_decoder + b_decoder;
    mse_relu(h) = (1/n) * norm(X - X_hat, 'fro')^2;
end

% Plot MSE vs h for the ReLU-based autoencoder
figure;
plot(1:6, mse_relu, '-o');
title('ReLU Autoencoder: MSE vs Code Dimension h');
xlabel('Code Dimension (h)');
ylabel('Mean Squared Error (MSE)');
grid on;

