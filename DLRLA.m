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

%% Step 1: Low-Order Feature-Label Relevance using pinv with Lambda1 and Lambda2
disp('Computing Low-Order FL Relevance using pinv with Lambda1 and Lambda2...');

% Regularization parameters
Lambda1 = 0.1;  % Reduced L1 regularization parameter
Lambda2 = 0.1;  % Reduced L2 regularization parameter

% Number of features and labels
num_features = size(X, 2);  % Number of features
num_labels = size(y, 2);    % Number of labels

% Add intercept term to the feature matrix
X_low = [ones(size(X, 1), 1), X];  % Add a column of ones for the intercept

% Compute the L2 term (Ridge Regression)
I = eye(size(X_low, 2));  % Identity matrix
L2_term = pinv(X_low' * X_low + Lambda2 * I) * X_low' * y;

% Approximate the L1 term (Lasso regularization) using soft-thresholding
% Soft-thresholding formula: D = sign(L2_term) .* max(abs(L2_term) - Lambda1, 0)
D_low = sign(L2_term) .* max(abs(L2_term) - Lambda1, 0);

% Extract coefficients for features only (exclude the intercept term)
low_order = D_low(2:end, :);  % Remove the first row (intercept coefficients)

% Display the low-order coefficients
disp('Low-Order Coefficients (excluding intercept):');
disp(low_order);


%% Step 2.1: Generate Candidate Feature Sets
disp('Generating Candidate Feature Sets...');

% Get the number of features
num_features = size(X, 2);

% Generate all possible candidate feature pairs (order-2)
candidate_pairs = nchoosek(1:num_features, 2);

% Display the candidate pairs
disp('Candidate Feature Pairs:');
disp(candidate_pairs);
% 
%% Step 2.2: Redundancy Computation using Tversky Similarity
disp('Computing Redundancy for Candidate Pairs using Tversky Similarity (Vectorized)...');

% Parameters for Tversky Similarity
alpha = 0.5;  % Weight for A \ B
beta = 0.5;   % Weight for B \ A

% Number of candidate pairs
num_pairs = size(candidate_pairs, 1);

% Extract feature vectors for all candidate pairs
F1 = X(:, candidate_pairs(:, 1));  % Features for the first element of each pair
F2 = X(:, candidate_pairs(:, 2));  % Features for the second element of each pair

% Compute intersection and differences
intersection = sum(min(F1, F2), 1);  % |A ? B| for each pair
F1_minus_F2 = sum(max(F1 - F2, 0), 1);  % |A \ B| for each pair
F2_minus_F1 = sum(max(F2 - F1, 0), 1);  % |B \ A| for each pair

% Compute Tversky Similarity for all pairs
tversky_similarity = intersection ./ (intersection + alpha * F1_minus_F2 + beta * F2_minus_F1);

% Store the redundancy values
redundancy_values = tversky_similarity';

% Display the redundancy values for each candidate pair
disp('Redundancy Values for Candidate Pairs:');
disp([candidate_pairs, redundancy_values]);

%% Step 2.3: Remove Redundant Pairs
disp('Removing Redundant Pairs...');

% Redundancy threshold
R_threshold = 0.4;

% Filter out pairs with redundancy > threshold
filtered_pairs = candidate_pairs(redundancy_values <= R_threshold, :);
filtered_redundancy_values = redundancy_values(redundancy_values <= R_threshold);

% Display the filtered pairs and their redundancy values
disp('Filtered Candidate Pairs (Redundancy <= 0.7):');
disp([filtered_pairs, filtered_redundancy_values]);


%% Step 2.4: Compute ?_high (High-Order Feature-Label Relevance)
disp('Computing High-Order FL Relevance using Elastic Net Regression...');

% Regularization parameters
Lambda1 = 0.01;  % Reduced L1 regularization parameter
Lambda2 = 0.01;  % Reduced L2 regularization parameter

% Initialize the high-order relevance matrix
delta_high = zeros(size(filtered_pairs, 1), num_class);

% Compute interaction-based relevance for each filtered pair and label
for i = 1:size(filtered_pairs, 1)
    % Get the indices of the current feature pair
    f1_idx = filtered_pairs(i, 1);
    f2_idx = filtered_pairs(i, 2);
    
    % Compute the interaction term (Fi × Fk)
    interaction_term = X(:, f1_idx) .* X(:, f2_idx);
    
    % Add the interaction term to the feature matrix
    X_high = [X, interaction_term];
    
    % Add intercept term
    X_high = [ones(size(X_high, 1), 1), X_high];
    
    % Compute the L2 term (Ridge Regression)
    I = eye(size(X_high, 2));  % Identity matrix
    L2_term = pinv(X_high' * X_high + Lambda2 * I) * X_high' * y;
    
    % Approximate the L1 term (Lasso regularization) using soft-thresholding
    % Soft-thresholding formula: D = sign(L2_term) .* max(abs(L2_term) - Lambda1, 0)
    D_high = sign(L2_term) .* max(abs(L2_term) - Lambda1, 0);
    
    % Extract coefficients for the interaction term only
    delta_high(i, :) = D_high(end, :);  % Interaction term is the last feature
end

% Display the high-order relevance matrix ?_high
disp('High-Order Relevance Matrix ?_high:');
disp(delta_high);

%% Step 3: Normalize ?_low and ?_high
disp('Normalizing ?_low and ?_high...');

% Normalize ?_low
max_delta_low = max(low_order(:));  % Find the maximum value in ?_low
delta_low_norm = low_order / max_delta_low;  % Normalize ?_low

% Normalize ?_high
max_delta_high = max(delta_high(:));  % Find the maximum value in ?_high
delta_high_norm = delta_high / max_delta_high;  % Normalize ?_high

% Display the normalized matrices
disp('Normalized Low-Order Relevance Matrix ?_low_norm:');
disp(delta_low_norm);

disp('Normalized High-Order Relevance Matrix ?_high_norm:');
disp(delta_high_norm);


%% Step 4: Combine Relevance Scores
disp('Combining Relevance Scores...');

% Initialize the combined relevance matrix
delta_combined = zeros(size(delta_low_norm));

% Iterate over all features and labels
for i = 1:size(delta_low_norm, 1)  % Loop over features
    for j = 1:size(delta_low_norm, 2)  % Loop over labels
        % Get the low-order relevance for feature F_i and label L_j
        low_relevance = delta_low_norm(i, j);
        
        % Find all high-order relevance values for pairs containing F_i and label L_j
        high_relevance = 0;
        for k = 1:size(filtered_pairs, 1)
            if filtered_pairs(k, 1) == i || filtered_pairs(k, 2) == i
                high_relevance = high_relevance + delta_high_norm(k, j)^2;
            end
        end
        
        % Combine the relevance scores
        delta_combined(i, j) = sqrt(low_relevance^2 + high_relevance);
    end
end

% Display the combined relevance matrix
disp('Combined Relevance Matrix ?_combined:');
disp(delta_combined);

%% Step 5: Compute Label Ambiguity (Entropy) -
disp('Computing Label Ambiguity using Entropy ..');

% Number of labels
num_labels = size(y, 2);

% Compute entropy for all labels
label_ambiguity = zeros(1, num_labels);  % Initialize the label ambiguity vector

% Compute the probability distribution for each label
for j = 1:num_labels
    % Get the current label vector
    label_vector = y(:, j);
    
    % Compute the probability distribution of the label
    unique_classes = unique(label_vector);  % Unique classes in the label
    class_counts = histcounts(label_vector, [unique_classes; max(unique_classes)+1]);  % Count occurrences of each class
    class_probabilities = class_counts / length(label_vector);  % Compute probabilities
    
    % Compute entropy using vectorized operations
    entropy = -sum(class_probabilities .* log2(class_probabilities + eps));  % Add eps to avoid log(0)
    
    % Store the entropy (label ambiguity)
    label_ambiguity(j) = entropy;
end

% Display the label ambiguity vector
disp('Label Ambiguity (Entropy):');
disp(label_ambiguity);


%% Step 6: Compute Quasi-Relevance Matrix M
disp('Computing Quasi-Relevance Matrix M...');

% Multiply each column of delta_combined by the corresponding label ambiguity entropy
Quasi = delta_combined .* label_ambiguity;

% Display the quasi-relevance matrix M
disp('Quasi-Relevance Matrix M:');
disp(Quasi);


%% Step 6: Apply GRO for Feature Ranking
disp('Applying Grey Relational Optimization (GRO) for Feature Ranking...');

W = label_ambiguity / sum(label_ambiguity);  % Normalize weights to sum to 1

% Step 1: Aspired and Worst Values
Quasi_star = max(Quasi);  % Aspired values (maximum for each label)
Quasi_min = min(Quasi);   % Worst values (minimum for each label)

% Step 2: Grey Relational Coefficients
lambda = 0.5;  % Distinguishing coefficient

% Initialize coefficient matrices
eta_star = zeros(size(Quasi));  % Coefficients for aspired values
eta_min = zeros(size(Quasi));   % Coefficients for worst values

% Compute grey relational coefficients
for i = 1:size(Quasi, 1)
    for j = 1:size(Quasi, 2)
        % Aspired coefficients
        delta_star = abs(Quasi_star(j) - Quasi(i, j));
        eta_star(i, j) = (min(min(abs(Quasi_star - Quasi))) + lambda * max(max(abs(Quasi_star - Quasi)))) / ...
                         (delta_star + lambda * max(max(abs(Quasi_star - Quasi))));
        
        % Worst coefficients
        delta_min = abs(Quasi_min(j) - Quasi(i, j));
        eta_min(i, j) = (min(min(abs(Quasi_min - Quasi))) + lambda * max(max(abs(Quasi_min - Quasi)))) / ...
                        (delta_min + lambda * max(max(abs(Quasi_min - Quasi))));
    end
end

% Step 3: Final Ranking
% Aggregate coefficients for Quasi_star and Quasi_min
eta_star_agg = sum(eta_star .* W, 2);  % Weighted sum for aspired coefficients
eta_min_agg = sum(eta_min .* W, 2);    % Weighted sum for worst coefficients

% Combine ranks
phi = eta_min_agg ./ (eta_star_agg + eta_min_agg);

% Display the final ranks
disp('Final Ranks (?_i):');
disp(phi);

% Rank the features based on ?_i (higher values are better)
[~, ranked_features] = sort(phi, 'descend');
disp('Ranked Features (from highest to lowest ?_i):');
disp(ranked_features');

toc;