function mean_best_score = mainGGO(function_name)
    num_runs = 51;  % Number of runs
    best_scores = zeros(1, num_runs);  % Array to store best score of each run

    % Run GGO algorithm 30 times
    for run = 1:num_runs
        best_scores(run) = runGGO2(function_name);
        fprintf('Run %d: Best score = %.6f\n', run, best_scores(run));
    end

    % Calculate mean of best scores over all runs
    mean_best_score = mean(best_scores);
    fprintf('Mean best score for %s over %d runs: %.6f\n', function_name, num_runs, mean_best_score);
end
