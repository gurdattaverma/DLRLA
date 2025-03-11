clc;
clear;

% Add dataset path
addpath("../dataset");

%% Load Data
dataset_name = input("Enter database Name: ", 's');
tic;
load("../dataset/" + dataset_name);
X = data;
f_count = size(data, 2); % Number of features
y = transpose(target);   % Labels
num_class = size(y, 2);  % Number of labels

% Main script
noofLabels = size(y, 2);

% Main script

noofLabels = size(y, 2);
% Main script


noofLabels = size(y, 2);

% PSO parameters
lb = [1, 0]; % Lower bounds for Neighbour and Smooth
ub = [20, 10]; % Upper bounds for Neighbour and Smooth
options = optimoptions('particleswarm', ...
    'SwarmSize', 20, ...
    'MaxIterations', 50, ...
    'FunctionTolerance', 1e-4, ...
    'Display', 'iter');
nvars = 2;

% Run PSO optimization
[bestParams, bestScore] = particleswarm(@(params) objectiveFunction(params, X, y), nvars, lb, ub, options);

% Display the best parameters and score
bestNeighbour = round(bestParams(1));
bestSmooth = bestParams(2);
fprintf('Best Neighbour: %d\n', bestNeighbour);
fprintf('Best Smooth: %.4f\n', bestSmooth);
fprintf('Best Mean Error: %.4f\n', bestScore);
tic
% Use the best parameters to evaluate the model
[X_train, Y_train, X_test, Y_test] = splitData(X, y);
[Prior, PriorN, Cond, CondN] = MLKNN_train(X_train, Y_train, bestNeighbour, bestSmooth);
[Outputs, Pre_Labels] = MLKNN_test(X_train, Y_train, X_test, bestNeighbour, Prior, PriorN, Cond, CondN);
accuracy = calculateAccuracy(Pre_Labels, Y_test);
fprintf('Final Accuracy with Best Parameters: %.4f\n', accuracy);
toc
% Objective function for PSO
function score = objectiveFunction(params, X, y)
    Neighbour = round(params(1)); % Round to nearest integer
    Smooth = params(2);
    
    if Neighbour < 1
        Neighbour = 1;
    end
    
    % 5-fold cross-validation
    cv = cvpartition(size(X,1),'KFold',5);
    
    total_error = 0;
    
    for fold = 1:cv.NumTestSets
        % Get train and test indices for this fold
        trainIdx = cv.training(fold);
        testIdx = cv.test(fold);
        
        % Split data
        X_train = X(trainIdx,:);
        Y_train = y(trainIdx,:);
        X_test = X(testIdx,:);
        Y_test = y(testIdx,:);
        
        % Train MLKNN
        [Prior, PriorN, Cond, CondN] = MLKNN_train(X_train, Y_train, Neighbour, Smooth);
        
        % Test MLKNN
        [Outputs, Pre_Labels] = MLKNN_test(X_train, Y_train, X_test, Neighbour, Prior, PriorN, Cond, CondN);
        
        % Calculate accuracy for this fold
        accuracy = calculateAccuracy(Pre_Labels, Y_test);
        
        % Accumulate error
        total_error = total_error + (1 - accuracy);
    end
    
    % Calculate mean error rate
    mean_error = total_error / cv.NumTestSets;
    
    % Return mean error rate for minimization
    score = mean_error;
end

% Function to split data into training and testing sets
function [X_train, Y_train, X_test, Y_test] = splitData(X, y)
    cv = cvpartition(size(X,1),'HoldOut',0.4);
    idx = cv.test;
    X_train = X(~idx,:);
    Y_train = y(~idx,:);
    X_test = X(idx,:);
    Y_test = y(idx,:);
end

% MLKNN training function
function [Prior, PriorN, Cond, CondN] = MLKNN_train(X_train, Y_train, Num, Smooth)
    [num_samples, num_features] = size(X_train);
    [num_labels, ~] = size(Y_train');
    
    % Calculate priors
    label_counts = sum(Y_train, 1); % Sum across samples for each label
    Prior = (label_counts + Smooth) ./ (num_samples + 2 * Smooth);
    PriorN = 1 - Prior;
    
    % Initialize conditional probabilities
    Cond = zeros(num_labels, Num + 1);
    CondN = zeros(num_labels, Num + 1);
    
    % Placeholder for conditional probability calculations
    % In practice, calculate based on nearest neighbor counts
    Cond = rand(num_labels, Num + 1);
    CondN = rand(num_labels, Num + 1);
    
    % Normalize conditional probabilities
    for i = 1:num_labels
        Cond(i,:) = Cond(i,:) / sum(Cond(i,:));
        CondN(i,:) = CondN(i,:) / sum(CondN(i,:));
    end
end

% MLKNN testing function
function [Outputs, Pre_Labels] = MLKNN_test(X_train, Y_train, X_test, Num, Prior, PriorN, Cond, CondN)
    [num_test, ~] = size(X_test);
    [num_labels, ~] = size(Y_train');
    
    % Placeholder for prediction logic
    % In practice, implement k-NN prediction
    Outputs = rand(num_labels, num_test);
    Pre_Labels = (Outputs >= 0.5)'; % Transpose to match Y_test dimensions
end

% Function to calculate accuracy
function accuracy = calculateAccuracy(Pre_Labels, Y_test)
    % Ensure dimensions match
    if ~isequal(size(Pre_Labels), size(Y_test))
        error('Dimension mismatch: Pre_Labels and Y_test must have the same size.');
    end
    
    % Calculate accuracy
    accuracy = mean(Pre_Labels == Y_test, 'all');
end