clear; clc; close all;

% Get the current script path
script_path = fileparts(mfilename('fullpath'));

if ~isempty(script_path)
    cd(script_path);
end

disp(['Working directory set to: ' pwd]);

path_healthy = fullfile('dataset', 'no');
path_pathological = fullfile('dataset', 'yes');

% Check if dataset folders exist
if ~isfolder(path_healthy) || ~isfolder(path_pathological)
    error('Error: Dataset folders not found. Please create "dataset/no" and "dataset/yes".');
end

%% TRAINING PHASE (SUPERVISED MACHINE LEARNING)
disp('--- Starting SVM Model Training ---');

training_features = []; % Empty matrix, row = patient and column = pixel
training_labels = []; % Boolean column vector of correct labels
target_size = [64, 64];

% Load HEALTHY (Label 0) and PATHOLOGICAL (Label 1) images
dataset_folders = {path_healthy, path_pathological};
labels_code = [0, 1]; % 0 = Healthy, 1 = Pathological

for k = 1:2
    curr_path = dataset_folders{k}; % Create a list with folder paths
    curr_label = labels_code(k); % k=1: Reads healthy, assigns label 0. k=2: Reads pathological, assigns label 1.
    
    % Search for various image formats
    f_list = dir(fullfile(curr_path, '*.jpg'));
    if isempty(f_list), f_list = dir(fullfile(curr_path, '*.png')); end
    if isempty(f_list), f_list = dir(fullfile(curr_path, '*.jpeg')); end
    
    fprintf('Loading %d images from: %s\n', length(f_list), curr_path);
    
    for i = 1:length(f_list)
        fname = fullfile(curr_path, f_list(i).name);
        img = imread(fname); % Load image
        
        % Pre-processing identical to the single case
        if size(img, 3) == 3, img = rgb2gray(img); end % If the image is RGB (3 channels), flatten it to grayscale (1 channel). 
        img = im2double(img); % Normalize image values
        img = imresize(img, target_size); % Resize image
        
        feat_vector = img(:)'; % Flattening: transform the 64x64 matrix into a 1x4096 row
        
        training_features = [training_features; feat_vector]; % Add a new row to the table (each image = row)
        training_labels = [training_labels; curr_label]; % Add the boolean value for classification
    end
end

%% --- TRAINING / TEST SPLIT ---
if isempty(training_features)
    error('No images found for training!');
end

disp('--- Dataset Splitting (Cross-Validation) ---');
% Set the seed to make the split repeatable (same results every time)
rng(42); % Fix the internal randomness of cvpartition

% Create the partition: 30% of data reserved for Testing
cv = cvpartition(training_labels, 'HoldOut', 0.3);

% Split features into two groups
X_train = training_features(training(cv), :); % 70% to study (Train)
Y_train = training_labels(training(cv), :);

X_test  = training_features(test(cv), :); % 30% for the final evaluation (Test)
Y_test  = training_labels(test(cv), :);

fprintf('Total images: %d\n', length(training_labels));
fprintf('TRAINING SET (Study): %d images\n', length(Y_train));
fprintf('TEST SET (Verification):   %d images\n', length(Y_test));

% --- TRAINING ---
disp('Training SVM...');
svm_model = fitcsvm(X_train, Y_train, 'KernelFunction', 'linear', 'Standardize', true);
disp('SVM model trained.');

% --- ACCURACY EVALUATION (Confusion Matrix) ---
disp('Verifying accuracy on the Test Set...');
predicted_labels_test = predict(svm_model, X_test); % Answers given by the SVM model on the test set

% Chart creation
figure('Name', 'Confusion Matrix', 'Color', 'w');
true_cats = categorical(Y_test, [0, 1], {'Healthy', 'Pathological'}); % Correct test answers
pred_cats = categorical(predicted_labels_test, [0, 1], {'Healthy', 'Pathological'}); % Model answers translated into readable labels

cm = confusionchart(true_cats, pred_cats); % Print the confusion matrix
cm.Title = 'Confusion Matrix'; 
cm.RowSummary = 'row-normalized'; % Sensitivity and Specificity (side table)
cm.ColumnSummary = 'column-normalized'; % Precision (bottom table)

accuracy_test = sum(predicted_labels_test == Y_test) / length(Y_test) * 100; % Calculate model accuracy
fprintf('REAL ACCURACY: %.2f%%\n', accuracy_test);
disp('------------------------------------------------');

%% REAL CASE SELECTION FROM THE TEST SET
rng('shuffle');
idx_test_random = randi(size(X_test, 1)); % Choose a random integer between 1 and the total number of test images

% Extract the pixel vector (the corresponding row)
img_flat_vector = X_test(idx_test_random, :);
true_label_val = Y_test(idx_test_random);

% Reconstruct the square image (Reshape from 1x4096 to 64x64)
img_input = reshape(img_flat_vector, target_size); % Exact inverse operation of img(:)

% Define the real status
if true_label_val == 1
    real_status = 'PATHOLOGICAL';
else
    real_status = 'HEALTHY';
end

fprintf('--- VISUAL DEMO ---\n');
fprintf('Case selected from Test Set (Index: %d)\n', idx_test_random);
fprintf('Real Status (Ground Truth): %s\n', real_status);

%% APPLICATION (SVM Prediction, PCA, and Spectral Clustering)

% --- MODEL PREDICTION (Machine Learning) ---
test_feature = img_input(:)'; % Flattening of the test image
predicted_label = predict(svm_model, test_feature);

if predicted_label == 1
    model_status = 'PATHOLOGICAL';
    color_status = 'r'; % Red if pathological
else
    model_status = 'HEALTHY';
    color_status = 'g'; % Green if healthy
end

fprintf('Model Prediction: %s\n', model_status);

% --- SEGMENTATION (Unsupervised Clustering) ---
% The code performs segmentation without knowing if it is healthy or sick.
k_clusters = 5; % k=5 to isolate a possible tumor.
[rows, cols] = size(img_input);
[X, Y] = meshgrid(1:cols, 1:rows); % Create two matrices (X for columns, Y for rows)
coords_norm = [(X(:)/cols)*0.5, (Y(:)/rows)*0.5]; % Normalize the coordinates by implementing them in a single matrix

% --- PCA (Linear Baseline) ---
features_pca = [img_input(:), coords_norm]; % For each pixel we create a vector of three numbers
[~, score, ~] = pca(features_pca); % Apply PCA on the matrix, returning only the score
rng(42); % Deterministic random generator
idx_pca = kmeans(score(:, 1:2), k_clusters, 'Replicates', 3); % Apply K-Means (we exclude the 3rd score column)
seg_pca = reshape(idx_pca, rows, cols); % We get the image we see in the final chart

% --- Spectral Clustering (Advanced Method) ---
sigma = 0.1; % Kernel sensitivity: controls how fast the similarity between two pixels decreases
alpha = 0.1; % Spatial weight: gives less importance to position and more importance to color
             % If alpha = 1 we would get the same error as PCA 
             % (vertical geometric cuts)
[seg_spectral, ~] = run_spectral_clustering(img_input, k_clusters, sigma, alpha);

%% VISUALIZATION

figure('Name', 'Segmentation & ML Analysis', 'Color', 'w', 'Position', [50, 50, 1000, 800]);

subplot(3, 3, 1); 
imshow(img_input); 
title({'1. Input Image', ['Real: ' real_status]}, 'FontSize', 10);

subplot(3, 3, 2); 
imagesc(seg_pca); axis image; axis off;
title('2. PCA Segmentation', 'FontSize', 12); colormap(gca, 'jet');

subplot(3, 3, 3); 
imagesc(seg_spectral); axis image; axis off;
title('3. Spectral Segmentation', 'FontSize', 12); 

for i = 1:k_clusters
    subplot(3, 3, 3+i);
    % Create a binary mask for cluster i
    mask = (seg_spectral == i);
    imshow(mask); 
    
    % Average intensity visualization
    mean_val = mean(img_input(mask));
    title(['Cluster ', num2str(i), ' (Intensity: ', num2str(mean_val, '%.2f'), ')']);
end

% --- Model Outcome Visualization ---
subplot(3,3,9); 
axis off;
text(0.0, 0.7, 'AI DIAGNOSIS (SVM):', 'FontSize', 12, 'FontWeight', 'bold');
text(0.0, 0.5, model_status, 'Color', color_status, 'FontSize', 14, 'FontWeight', 'bold');
text(0.0, 0.3, ['(Real: ' real_status ')'], 'Color', 'k', 'FontSize', 10);

disp('------------------------------------------------');
disp('Processing completed.');
disp('------------------------------------------------');

%% LOCAL FUNCTION
function [segmentation, eig_vectors] = run_spectral_clustering(img, k, sigma, alpha)
    [rows, cols] = size(img);
    N = rows * cols; % Total number of nodes
    [X, Y] = meshgrid(1:cols, 1:rows); % Creation of matrices with X and Y coordinates
    feat_coords = [X(:)/cols, Y(:)/rows]; % Coordinate normalization
    feat_int = img(:);
    features = [feat_int, alpha * feat_coords]; % 3-element vector
    
    sum_sq = sum(features.^2, 2);
    dist_sq = sum_sq + sum_sq' - 2 * (features * features'); % Squared Euclidean distance
    W = exp(-dist_sq / (2 * sigma^2)); % Affinity matrix (The Graph)
    W(1:N+1:end) = 0; % Set the diagonal to 0
    W(W < 1e-4) = 0; % Zero out weak connections
    W = sparse(W); % Sparse matrix for efficiency (storing non-zero values)
    
    D_vec = sum(W, 2); % Degree matrix D (sum of W rows)
    D_inv_sqrt = spdiags(1./sqrt(D_vec + eps), 0, N, N); % Division by zero protection with eps
    L_sym = speye(N) - D_inv_sqrt * W * D_inv_sqrt; % Normalized Laplacian
    
    [eig_vectors, ~] = eigs(L_sym, k, 'sa'); % k smallest eigenvectors
    rng(42);
    idx = kmeans(eig_vectors, k, 'Replicates', 3); % Apply K-Means on the rows of the eigenvectors matrix
    segmentation = reshape(idx, rows, cols); % Image reconstruction
end