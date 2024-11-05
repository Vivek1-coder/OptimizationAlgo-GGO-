% main.m
clear;
clc;

% List of functions to evaluate
function_names = {'F1','F3','F4','F5','F6','F7','F8','F9','F10','F11','F12','F13','F14','F15','F16','F17','F18','F19','F20','F21','F22','F23','F24','F25','F26','F27','F28','F29','F30'};

% Preallocate array to store mean best scores for each function
mean_best_scores = zeros(length(function_names), 1);

% Run GGO for each function and store the mean best score
for i = 1:length(function_names)
    mean_best_scores(i) = runGGO2(function_names{i});
end

% Display mean best scores for each function
for i = 1:length(function_names)
    fprintf('Mean best score for %s: %.6f\n', function_names{i}, mean_best_scores(i));
end
