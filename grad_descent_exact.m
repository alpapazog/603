% Define the objective function
function f_val = f(x, P, q)
    log_term = log(exp(-2 * x(1)) + exp(-x(2)));
    f_val = 0.5 * x' * P * x + q' * x + log_term;
end

% Define the gradient of the objective function
function grad = grad_f(x, P, q)
    exp_term1 = exp(-2 * x(1));
    exp_term2 = exp(-x(2));
    denom = exp_term1 + exp_term2;
    grad_log_term = [-2 * exp_term1 / denom;
                      -exp_term2 / denom];
    grad = P * x + q + grad_log_term;
end

% Define the derivative of the line search objective
function dphi = dphi(alpha, x, P, q, d)
    grad_at_alpha = grad_f(x + alpha * d, P, q);
    dphi = grad_at_alpha' * d;  % Compute the directional derivative
end

% Exact line search using fminbnd with derivative-based search
function alpha = exact_line_search_with_derivative(x, P, q, d)
    % Define the objective function for line search
    phi = @(alpha) f(x + alpha * d, P, q);
    
    % Define the derivative of phi
    dphi_handle = @(alpha) dphi(alpha, x, P, q, d);
    
    % Use fminbnd or any optimization method that can handle derivatives
    options = optimset('GradObj', 'on', 'TolX', 1e-6);
    
    % Use fminunc to perform minimization with derivative
    alpha = fminunc(@(alpha) deal(phi(alpha), dphi_handle(alpha)), 0, options);
end

% Gradient descent algorithm with exact line search using derivative
function [x_vals] = gradient_descent_exact(P)
    % Initialize parameters
    %P = [3 4; 4 6];
    %P = [5.005 4.995; 4.995 5.005];
    q = [-2; 4];
    x0 = [1; 2];
    tol = 1e-2;  % Stopping criterion for gradient norm
    x_k = x0;
    grad_norms = [];
    x_vals = x_k;
    % Perform gradient descent
    while norm(grad_f(x_k, P, q)) >= tol
        d_k = -grad_f(x_k, P, q);
        alpha_k = exact_line_search_with_derivative(x_k, P, q, d_k);
        x_k = x_k + alpha_k * d_k;
        x_vals = [x_vals, x_k]; % Store solution sequence
        grad_norms = [grad_norms, norm(grad_f(x_k, P, q))];
    end
    
    figure;
    plot(x_vals(1,:), x_vals(2,:), '-o', 'LineWidth', 2);
    xlabel('x_1');
    ylabel('x_2');
    title('Solutions using Gradient Descent with Exact Line Search');
    grid on;
end

[x_vals_exact1] = gradient_descent_exact([3 4; 4 6]);
[x_vals_exact2] = gradient_descent_exact([5.005 4.995; 4.995 5.005]);
