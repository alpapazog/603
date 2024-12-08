trainImages = loadIDX('train-images.idx3-ubyte');
trainLabels = loadIDX('train-labels.idx1-ubyte');
testImages = loadIDX('t10k-images.idx3-ubyte');
testLabels = loadIDX('t10k-labels.idx1-ubyte');

% Normalize images to [0, 1]
trainImages = trainImages / 255;
testImages = testImages / 255;

% Reshape the data for neural network input
trainImages = reshape(trainImages, [28*28, size(trainImages, 3)])'; % Each row is an image
testImages = reshape(testImages, [28*28, size(testImages, 3)])';

% Convert labels to categorical for classification
trainLabels = categorical(trainLabels);
testLabels = categorical(testLabels);

layers = [
    featureInputLayer(28*28, 'Name', 'input') % Input layer for 28x28 images
    fullyConnectedLayer(128, 'Name', 'fc1')   % First hidden layer with 128 nodes
    reluLayer('Name', 'relu1')                % ReLU activation
    fullyConnectedLayer(64, 'Name', 'fc2')    % Second hidden layer with 64 nodes
    reluLayer('Name', 'relu2')                % ReLU activation
    fullyConnectedLayer(10, 'Name', 'fc3')    % Output layer with 10 nodes (for 10 digits)
    softmaxLayer('Name', 'softmax')           % Softmax activation
    classificationLayer('Name', 'output')     % Classification layer
];

options = trainingOptions('adam', ... % Use the Adam optimizer
    'InitialLearnRate', 0.001, ...    % Learning rate
    'MaxEpochs', 10, ...             % Number of epochs
    'MiniBatchSize', 128, ...        % Batch size
    'Shuffle', 'every-epoch', ...    % Shuffle data every epoch
    'ValidationData', {testImages, testLabels}, ... % Validation set
    'ValidationFrequency', 30, ...  % Validate every 30 iterations
    'Plots', 'none', ...            % No training progress
    'Verbose', false);               % Disable verbose output


numRuns = 5;  % Number of training runs
accuracies = zeros(numRuns, 1);  % Preallocate for accuracies

for i = 1:numRuns
    % Train the network
    net = trainNetwork(trainImages, trainLabels, layers, options);
    
    % Test the network
    predictedLabels = classify(net, testImages);
    
    % Calculate accuracy
    accuracy = sum(predictedLabels == testLabels) / numel(testLabels);
    accuracies(i) = accuracy;
    
    fprintf('Run %d Accuracy: %.2f%%\n', i, accuracy * 100);
end

% Display overall results
fprintf('Average Accuracy: %.2f%%\n', mean(accuracies) * 100);
fprintf('Accuracy Standard Deviation: %.2f%%\n', std(accuracies) * 100);