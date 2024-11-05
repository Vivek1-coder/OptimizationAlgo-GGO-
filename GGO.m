% GGO.m
function ggo_params = GGO(num_agents, max_iterations)
    % Initialize GGO parameters according to exploration and exploitation details
    
    % General parameters
    ggo_params.num_agents = num_agents;
    ggo_params.max_iterations = max_iterations;
    
    % Parameters that change during iterations
    ggo_params.a = 2;          % 'a' starts at 2 and decreases linearly to 0
    ggo_params.b = 1.5;        % 'b' constant for cosine term in exploitation
    ggo_params.l = 0.5;        % Random value in [-1, 1]
    ggo_params.c = 0.8;        % Additional coefficient for certain terms
    
    % Initial weights for paddling and sentry steps
    ggo_params.w1 = 0.6;
    ggo_params.w2 = 0.4;
    ggo_params.w3 = 0.3;
    ggo_params.w4 = 0.2;
    % Random factors `r` (these are recalculated at each iteration)
    ggo_params.r1 = rand(num_agents, 1);
    ggo_params.r2 = rand(num_agents, 1);
    ggo_params.r3 = rand(num_agents, 1); 
    ggo_params.r4 = rand(num_agents, 1);
    ggo_params.r5 = rand(num_agents, 1);
    
    % Calculating vectors A, C for the current iteration
    % For A and C, a separate update for each agent during each iteration is recommended.
    ggo_params.A = @(a, r1) 2 * a * r1 - a;   % Calculate A as per iteration with updated 'a'
    ggo_params.C = @(r2) 2 * r2;              % Calculate C using random r2
    
    % Calculation for A1, A2, A3 and C1, C2, C3 in exploitation (updated per iteration)
    ggo_params.A1 = @(a, r1) 2 * a * r1 - a;
    ggo_params.A2 = @(a, r1) 2 * a * r1 - a;
    ggo_params.A3 = @(a, r1) 2 * a * r1 - a;
    
    ggo_params.C1 = @(r2) 2 * r2;
    ggo_params.C2 = @(r2) 2 * r2;
    ggo_params.C3 = @(r2) 2 * r2;
    % Exponential decay for z, calculated per iteration
    ggo_params.z = 1;  % Initial value, will be updated in runGGO

    % Weights updating will occur during each iteration
    ggo_params.update_weights = @(t) deal( ...
        min(2, max(0, 2 * (1 - t / max_iterations) + ggo_params.w1)), ...  % w1 updates
        min(2, max(0, 2 * (1 - t / max_iterations) + ggo_params.w2)), ...  % w2 updates
        min(2, max(0, 2 * (1 - t / max_iterations) + ggo_params.w3)), ...  % w3 updates
        min(2, max(0, 2 * (1 - t / max_iterations) + ggo_params.w4)) ...   % w4 updates
    );
end
