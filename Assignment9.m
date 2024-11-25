% Load the dataset
data = readtable('moonDataset.csv');
features = data{:, 1:3};  % features
labels = data{:, 4};      % label (0 or 1)

% Partition the data into training (150 samples) and testing (50 samples)
train_features = features(1:150, :);
train_labels = labels(1:150);
test_features = features(151:end, :);
test_labels = labels(151:end);


% (a) Bootstrap Sampling

% Number of bootstrap datasets
nBootstrap = 50;
bootstrap_datasets = cell(nBootstrap, 1);

for i = 1:nBootstrap
    % Generate a bootstrap sample (random sampling with replacement)
    bootstrap_indices = randi(150, 150, 1);  % Random indices with replacement
    bootstrap_datasets{i}.features = train_features(bootstrap_indices, :);
    bootstrap_datasets{i}.labels = train_labels(bootstrap_indices);
end


% (b) Constructing and Training Neural Networks

% Store error rates
error_rates = zeros(nBootstrap, 1);

for i = 1:nBootstrap
    % Get bootstrap dataset
    train_data = bootstrap_datasets{i};
    
    % Create a feedforward neural network with 10 hidden nodes
    net = feedforwardnet(10);

    % Suppress the training window
    net.trainParam.showWindow = false;
    
    % Apply L2 regularization (default is 0)
    net.performParam.regularization = 0.01;

    % Train the network
    net = train(net, train_data.features', train_data.labels');
    
    % Test the network
    predictions = net(test_features') > 0.5;  % Binary classification threshold
    error_rates(i) = mean(predictions ~= test_labels');  % Compute error rate
end

% Plot histogram of error rates
figure;
histogram(error_rates, 10);
title('Histogram of Error Rates');
xlabel('Error Rate');
ylabel('Frequency');

%(c) Bagging with Different Ensemble Sizes

ensemble_sizes = [5, 10, 15, 20];
ensemble_error_rates = zeros(length(ensemble_sizes), 1);

for j = 1:length(ensemble_sizes)
    m = ensemble_sizes(j);
    bagging_predictions = zeros(m, length(test_labels));
    
    % Create m models in the ensemble
    for k = 1:m
        % Select a bootstrap dataset
        bootstrap_indices = randi(150, 150, 1); 
        train_data = bootstrap_datasets{randi(nBootstrap)};  % Randomly select a bootstrap dataset
        
        % Create and train a neural network
        net = feedforwardnet(10);
        % Suppress the training window
        net.trainParam.showWindow = false;
        % Apply L2 regularization (default is 0)
        net.performParam.regularization = 0.01;
        net = train(net, train_data.features', train_data.labels');
        
        % Store predictions for this model
        bagging_predictions(k, :) = net(test_features') > 0.5;
    end
    
    % Perform majority voting across models in the ensemble
    final_predictions = mode(bagging_predictions, 1);
    
    % Calculate error rate
    ensemble_error_rates(j) = mean(final_predictions ~= test_labels');
end

% Plot the error rate as a function of ensemble size
figure;
plot(ensemble_sizes, ensemble_error_rates, '-o');
title('Error Rate vs. Ensemble Size');
xlabel('Ensemble Size (m)');
ylabel('Error Rate');
