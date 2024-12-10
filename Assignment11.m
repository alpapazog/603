% Load the dataset
data = readmatrix('HW11-ClusteringData.csv');

% Extract data points and true labels
X = data(:, 1:2); % First two columns: data points
labels = data(:, 3); % Third column: true labels

%a)
% Plot the data points with different colors for each label
figure;
gscatter(X(:,1), X(:,2), labels);
xlabel('X1');
ylabel('X2');
title('Data Points Colored by True Labels');
legend('Component 1', 'Component 2', 'Component 3', 'Component 4');
grid on;

%b)
% Initialize arrays to store average silhouette scores
k_values = 2:7;
avg_sil_euclidean = zeros(size(k_values));
avg_sil_manhattan = zeros(size(k_values));

% Loop over values of k
for i = 1:length(k_values)
    k = k_values(i);
    
    % K-means clustering with Euclidean distance
    [idx_euclidean, ~] = kmeans(X, k, 'Distance', 'sqeuclidean');
    
    % Compute silhouette scores for Euclidean distance
    sil_euclidean = silhouette(X, idx_euclidean, 'sqeuclidean');
    avg_sil_euclidean(i) = mean(sil_euclidean);
    
    % K-means clustering with Manhattan distance
    [idx_manhattan, ~] = kmeans(X, k, 'Distance', 'cityblock');
    
    % Compute silhouette scores for Manhattan distance
    sil_manhattan = silhouette(X, idx_manhattan, 'cityblock');
    avg_sil_manhattan(i) = mean(sil_manhattan);
end

% Plot the average silhouette coefficients
figure;
plot(k_values, avg_sil_euclidean, '-o', 'LineWidth', 2); hold on;
plot(k_values, avg_sil_manhattan, '-x', 'LineWidth', 2);
xlabel('Number of Clusters (k)');
ylabel('Average Silhouette Coefficient');
title('Average Silhouette Coefficient for Different k');
legend('Euclidean Distance', 'Manhattan Distance');
grid on;

% Find optimal k
[~, optimal_k_euclidean_idx] = max(avg_sil_euclidean);
optimal_k_euclidean = k_values(optimal_k_euclidean_idx);

[~, optimal_k_manhattan_idx] = max(avg_sil_manhattan);
optimal_k_manhattan = k_values(optimal_k_manhattan_idx);

fprintf('Optimal k (Euclidean): %d\n', optimal_k_euclidean);
fprintf('Optimal k (Manhattan): %d\n', optimal_k_manhattan);

%c)
N = size(X, 1);   % Number of data points
K = 4;            % Number of clusters

% Initialize parameters randomly
rng(1); % For reproducibility
pi_k = ones(1, K) / K; % Equal weights initially
mu_k = X(randperm(N, K), :); % Randomly select initial means
Sigma_k = repmat(eye(2), [1, 1, K]); % Identity matrices

% EM Algorithm
tol = 1e-6; % Tolerance for convergence
log_likelihood_prev = -inf;

for iter = 1:1000
    % E-Step: Compute responsibilities (gamma)
    gamma = zeros(N, K);
    for k = 1:K
        gamma(:, k) = pi_k(k) * mvnpdf(X, mu_k(k, :), Sigma_k(:, :, k));
    end
    gamma = gamma ./ sum(gamma, 2);

    % M-Step: Update parameters
    Nk = sum(gamma, 1);
    pi_k = Nk / N;
    for k = 1:K
        mu_k(k, :) = sum(gamma(:, k) .* X) / Nk(k);
        diff = X - mu_k(k, :);
        Sigma_k(:, :, k) = (diff' * (gamma(:, k) .* diff)) / Nk(k);
    end

    % Compute log-likelihood
    log_likelihood = sum(log(sum(gamma, 2)));
    if abs(log_likelihood - log_likelihood_prev) < tol
        break;
    end
    log_likelihood_prev = log_likelihood;
end

% Display estimated parameters
disp('Estimated mixing coefficients (pi_k):');
disp(pi_k);
disp('Estimated means (mu_k):');
disp(mu_k);

% Plot the Gaussian mixture
x = linspace(min(X(:, 1)), max(X(:, 1)), 100);
y = linspace(min(X(:, 2)), max(X(:, 2)), 100);
[X1, X2] = meshgrid(x, y);

% Initialize the pdf_values as a grid of zeros
pdf_values = zeros(size(X1));

for k = 1:K
    % Compute the PDF for each Gaussian component
    pdf_component = mvnpdf([X1(:), X2(:)], mu_k(k, :), Sigma_k(:, :, k));
    % Reshape the component PDF to match the grid size
    pdf_values = pdf_values + pi_k(k) * reshape(pdf_component, size(X1));
end
pdf_values = reshape(pdf_values, size(X1));

figure;
contour(X1, X2, pdf_values, 20); % Contour plot
hold on;
scatter(X(:, 1), X(:, 2), 10, 'filled'); % Data points
scatter(mu_k(:, 1), mu_k(:, 2), 100, 'rx', 'LineWidth', 2); % Estimated means
title('Gaussian Mixture Model - EM Algorithm');
xlabel('x1');
ylabel('x2');
grid on;

% Compare with true parameters
true_mu = [4, 6; -3, 3; 2, -2; -1, -7];
true_pi = [0.1875, 0.25, 0.3438, 0.2188];
disp('True means (mu_k):');
disp(true_mu);
disp('True mixing coefficients (pi_k):');
disp(true_pi);
