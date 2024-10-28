function [x_vals] = gradient_descent_backtrack(P)
    % Parameters
    q = [-2; 4];
    alpha_init = 0.15;
    gamma = 0.7;
    beta = 0.8;
    tol = 1e-2;
    x0 = [1; 2]; % Initial point
    max_iter = 5000;

    % Objective function
    f = @(x) 0.5 * x' * P * x + q' * x + log(exp(-2 * x(1)) + exp(-x(2)));
    
    % Gradient of the objective function
    grad_f = @(x) P * x + q + [(-2 * exp(-2 * x(1))) / (exp(-2 * x(1)) + exp(-x(2)));
                               (-exp(-x(2))) / (exp(-2 * x(1)) + exp(-x(2)))];
    
    % Gradient descent with backtracking line search
    x = x0;
    x_vals = x0; % To store the sequence of solutions
    for k = 1:max_iter
        g = grad_f(x);
        if norm(g) < tol
            break;
        end
        
        % Backtracking line search
        alpha = alpha_init;
        while f(x - alpha * g) > f(x) - gamma * alpha * norm(g)^2
            alpha = beta * alpha;
        end
        
        % Update x
        x = x - alpha * g;
        x_vals = [x_vals, x]; % Append current solution to history
    end
    k
    figure;
    plot(x_vals(1, :), x_vals(2, :), 'bo-', 'LineWidth', 1.5);
    xlabel('x_1');
    ylabel('x_2');
    title('Solutions using Gradient Descent with Backtracking Line Search');
    grid on;
end

[x_vals_back1] = gradient_descent_backtrack([3 4; 4 6]);
[x_vals_back2] = gradient_descent_backtrack([5.005 4.995; 4.995 5.005]);