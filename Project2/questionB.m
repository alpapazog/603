trainImages = loadIDX('train-images.idx3-ubyte');
trainLabels = loadIDX('train-labels.idx1-ubyte');
testImages = loadIDX('t10k-images.idx3-ubyte');
testLabels = loadIDX('t10k-labels.idx1-ubyte');

% Normalize the images
trainImages = double(trainImages) / 255;
testImages = double(testImages) / 255;

% Reshape the images to 4D arrays
trainImages = reshape(trainImages, [28, 28, 1, size(trainImages, 3)]);
testImages = reshape(testImages, [28, 28, 1, size(testImages, 3)]);

% Convert labels to categorical
trainLabels = categorical(trainLabels);
testLabels = categorical(testLabels);

layers = [
    imageInputLayer([28 28 1])               % Input layer for 28x28 grayscale images

    % First convolutional layer
    convolution2dLayer(3, 8, 'Padding', 'same') % 3x3 filter, 8 filters, same padding
    batchNormalizationLayer                   % Batch normalization
    reluLayer                                 % ReLU activation
    maxPooling2dLayer(2, 'Stride', 2)         % Max pooling with 2x2 filter and stride 2

    % Second convolutional layer
    convolution2dLayer(3, 16, 'Padding', 'same') % 3x3 filter, 16 filters, same padding
    batchNormalizationLayer                   % Batch normalization
    reluLayer                                 % ReLU activation
    maxPooling2dLayer(2, 'Stride', 2)         % Max pooling with 2x2 filter and stride 2

    % Fully connected layers
    fullyConnectedLayer(128)                  % First fully connected layer with 128 neurons
    reluLayer                                 % ReLU activation
    fullyConnectedLayer(10)                   % Output layer with 10 neurons for 10 classes
    softmaxLayer                              % Softmax for probability output
    classificationLayer                       % Classification layer
];

options = trainingOptions('adam', ...       % Adam optimizer
    'MaxEpochs', 10, ...                   % Number of epochs
    'MiniBatchSize', 128, ...              % Batch size
    'InitialLearnRate', 1e-3, ...          % Initial learning rate
    'Shuffle', 'every-epoch', ...          % Shuffle data every epoch
    'Verbose', false, ...                  % Suppress output
    'Plots', 'none');                       % Don't display training progress

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