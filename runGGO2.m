function best_score = runGGO2(function_name)
    % Load function details
    [lb, ub, dim, fobj] = CEC2017(function_name);
    max_iterations = 1000;      % Set initial max iterations (may stop earlier due to eval limit)
    num_agents = 50;            % Number of agents
    ggo_params = GGO(num_agents, max_iterations);

    % Initialize population positions
    positions = lb + (ub - lb) * rand(ggo_params.num_agents, dim);
    best_scores_over_time = zeros(1, ggo_params.max_iterations);  % For tracking progress

    % Initialize best scores
    best_personal_scores = arrayfun(@(idx) fobj(positions(idx, :)), 1:ggo_params.num_agents);
    [best_global_score, best_idx] = min(best_personal_scores);
    best_global_position = positions(best_idx, :);

    % Initialize function evaluation counter
    func_evals = ggo_params.num_agents;  % Initial evaluations from initializing `best_personal_scores`
    max_func_evals = 50000;              % Maximum allowed function evaluations

    for t = 1:ggo_params.max_iterations
        if func_evals >= max_func_evals
            break;  % Stop if function evaluations exceed limit
        end

        % Dynamic parameter update (non-linear decay)
        ggo_params.a = 2 - (2 * (t / ggo_params.max_iterations)^1.5);  % Non-linear decay
        ggo_params.z = exp(-3 * (t / ggo_params.max_iterations));      % Exponential decay for 'z'
        
        % Adaptive weights based on GWO hierarchy
        ggo_params.w1 = 0.5 + (1 - t / ggo_params.max_iterations);
        ggo_params.w2 = 0.5 + (0.5 * sin(pi * t / ggo_params.max_iterations));
        ggo_params.w3 = 0.3 + 0.2 * cos(2 * pi * t / ggo_params.max_iterations);
        ggo_params.w4 = 0.2 + 0.1 * (1 - cos(pi * t / ggo_params.max_iterations));

        % Select top 3 leaders (alpha, beta, delta) for the hierarchical structure
        [sorted_scores, sorted_idx] = sort(best_personal_scores);
        alpha_position = positions(sorted_idx(1), :);
        beta_position = positions(sorted_idx(2), :);
        delta_position = positions(sorted_idx(3), :);

        for i = 1:ggo_params.num_agents
            if func_evals >= max_func_evals
                break;  % Stop inner loop if function evaluations exceed limit
            end

            % Calculate A and C for exploration/exploitation
            A = ggo_params.A(ggo_params.a, rand); % Calculate A for this iteration
            C = ggo_params.C(rand);               % Calculate C
            
            % Exploration or exploitation based on `|A|` threshold
            if abs(A) >= 1
                % Exploration step with random walk if far from leaders
                random_factor = rand(dim, 1)';
                new_position = positions(i, :) + A * (random_factor .* (alpha_position - positions(i, :)) ...
                                 + (1 - random_factor) .* (beta_position - delta_position));
            else
                % Exploitation step influenced by leaders
                D_alpha = abs(C .* alpha_position - positions(i, :));
                D_beta = abs(C .* beta_position - positions(i, :));
                D_delta = abs(C .* delta_position - positions(i, :));

                X1 = alpha_position - A * D_alpha;
                X2 = beta_position - A * D_beta;
                X3 = delta_position - A * D_delta;

                new_position = (X1 + X2 + X3) / 3;
            end
            
            % Boundary check to keep within limits
            new_position = boundaryCheck(new_position, lb, ub);
            
            % Evaluate new position
            score = fobj(new_position);
            func_evals = func_evals + 1;  % Increment evaluation count

            % Update position and scores
            positions(i, :) = new_position;
            if score < best_personal_scores(i)
                best_personal_scores(i) = score;
            end
            if score < best_global_score
                best_global_score = score;
                best_global_position = positions(i, :);
            end
        end

        % Track best score per iteration for plotting
        best_scores_over_time(t) = best_global_score;
    end
    
    best_score = best_global_score;

    % Plot convergence curve after all iterations
    figure;
    semilogy(1:t, best_scores_over_time(1:t), '-o');
    title(['Convergence Curve for ', function_name]);
    xlabel('Iteration');
    ylabel('Best Score');
    grid on;
end

% Helper function for boundary checking
function positions = boundaryCheck(positions, lb, ub)
    positions = max(min(positions, ub), lb);
end
