function best_score = runGGO(function_name)
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
        
        % Dynamic parameter update
        ggo_params.a = 2 - (1.5 * t / ggo_params.max_iterations);  % Slower decay for exploration
        ggo_params.z = 1 - (t / ggo_params.max_iterations)^2;      % Exponential decay for 'z'
        
        % Update weights dynamically in range [0, 2]
        ggo_params.w1 = 2 * abs(sin(pi * t / ggo_params.max_iterations));
        ggo_params.w2 = 2 * abs(cos(pi * t / ggo_params.max_iterations));
        ggo_params.w3 = 1.5 * abs(sin(pi * t / (2 * ggo_params.max_iterations)));
        ggo_params.w4 = 1.5 * abs(cos(pi * t / (2 * ggo_params.max_iterations)));
        
        for i = 1:ggo_params.num_agents
            if func_evals >= max_func_evals
                break;  % Stop inner loop if function evaluations exceed limit
            end

            % Calculate A and C for exploration/exploitation
            A = ggo_params.A(ggo_params.a, rand); % Calculate A for this iteration
            C = ggo_params.C(rand);               % Calculate C
            
            if abs(A) >= 1
                % Exploration (paddling formula)
                paddles = randperm(ggo_params.num_agents, 3);
                Xpaddle1 = positions(paddles(1), :);
                Xpaddle2 = positions(paddles(2), :);
                Xpaddle3 = positions(paddles(3), :);
                
                positions(i, :) = ggo_params.w1 * Xpaddle1 + ggo_params.z * ggo_params.w2 * rand .* (Xpaddle2 - Xpaddle3) ...
                                  + (1 - ggo_params.z) * ggo_params.w3 * rand .* (best_global_position - Xpaddle1);
            else
                % Exploitation (sentry formula)
                sentries = randperm(ggo_params.num_agents, 3);
                Xsentry1 = positions(sentries(1), :);
                Xsentry2 = positions(sentries(2), :);
                Xsentry3 = positions(sentries(3), :);
                
                X1 = Xsentry1 - ggo_params.A1(ggo_params.a, rand) .* abs(ggo_params.C1(rand) .* Xsentry1 - positions(i, :));
                X2 = Xsentry2 - ggo_params.A2(ggo_params.a, rand) .* abs(ggo_params.C2(rand) .* Xsentry2 - positions(i, :));
                X3 = Xsentry3 - ggo_params.A3(ggo_params.a, rand) .* abs(ggo_params.C3(rand) .* Xsentry3 - positions(i, :));
                
                positions(i, :) = mean([X1; X2; X3], 1);
            end

            % Boundary check to keep within limits
            positions(i, :) = boundaryCheck(positions(i, :), lb, ub);

            % Evaluate new position
            score = fobj(positions(i, :));
            func_evals = func_evals + 1;  % Increment evaluation count

            % Update personal and global bests
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
    plot(1:t, best_scores_over_time(1:t), '-o');
    title(['Convergence Curve for ', function_name]);
    xlabel('Iteration');
    ylabel('Best Score');
    grid on;
end

% Helper function for boundary checking
function positions = boundaryCheck(positions, lb, ub)
    positions = max(min(positions, ub), lb);
end
