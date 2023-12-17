%% Preprocessing and Saving Images
% Specify the path to your dataset folder
datasetPath = './dataset/fractured/';
ins = './dataset/';

% List all image files in the dataset folder
filePaths = dir(fullfile(datasetPath, '*.jpg'));
edgeImages = cell(1, numel(filePaths));

for i = 1:length(filePaths)
    disp(filePaths(i).name);
end

% Load images and apply the Canny edge detector
for j = 1:numel(filePaths)
    % Load the original image
    originalImage = imread(fullfile(datasetPath, filePaths(j).name));
    
    % Convert the image to grayscale if it's an RGB image
    if size(originalImage, 3) == 3
        grayscaleImage = rgb2gray(originalImage);
    else
        grayscaleImage = originalImage;
    end
   
    % Display the original image
    figure;
    subplot(1, 5, 1);
    imshow(grayscaleImage);
    title(filePaths(j).name);

    %  Apply Gaussian filter for smoothing
    smoothedImage = imgaussfilt(grayscaleImage, 8);
    subplot(1, 5, 2);
    imshow(smoothedImage);
    title(' Guassian Filtered Image');

    % Enhance edges using image sharpening (e.g., unsharp masking)
    sharpenedImage = imsharpen(smoothedImage, 'Amount', 4);
    subplot(1, 5, 3);
    imshow(sharpenedImage);
    title(' Sharpened Image');

    % Contrast improvement using adaptive histogram equalization
    enhancedImage = adapthisteq(sharpenedImage);
    subplot(1, 5, 4);
    imshow(enhancedImage);
    title('Contrast Improved Image(w/Histogram');
    
     % Apply Canny edge detection on the enhanced image
    edges = edge(sharpenedImage, 'Canny');
    edgeImages{j} = edges;
    subplot(1, 5, 5);
    imshow(edges);
    title('Canny Edge Detection'); 
    
    %imwrite(edges, fullfile(ins, ['pre_' filePaths(j).name]));
end
%% SVM Classification
% Extract labels from the file names
imagePath = './dataset/';
filePaths = dir('./dataset/*.jpg');
labels = arrayfun(@(file) extractLabel(file.name), filePaths, 'UniformOutput', false);
% Convert labels to numeric format
numericLabels = str2double(labels);

% Initialize an empty cell array to store flattened images
flattenedImages = cell(1, numel(filePaths));

% Set a common size for resizing
commonSize = [100, 100]; % Adjust the size as needed

% Iterate through each image
for i = 1:numel(filePaths)
    % Load the preprocessed image
    preprocessedImage = imread(fullfile(imagePath, filePaths(i).name));
    % Resize the image to a common size
    resizedImage = imresize(preprocessedImage, commonSize);
    % Flatten the resized image into a row vector
    flattenedImage = resizedImage(:)';
    % Store the flattened image in the cell array
    flattenedImages{i} = flattenedImage;
end

% Check the length of the flattened images
imageLengths = cellfun(@length, flattenedImages);

% Ensure that there is at least one image and all images have the same length
if isempty(imageLengths) || ~all(imageLengths == imageLengths(1))
    error('Images have inconsistent lengths or no images are loaded.');
end

% Create the feature matrix by converting cell array to a matrix
featureMatrix = cell2mat(flattenedImages);

% Specify the ratio for training and testing data
trainRatio = 0.8;
testRatio = 1 - trainRatio;

% Create random indices for training and testing
numSamples = size(featureMatrix, 1);
idx = randperm(numSamples);

% Determine the number of samples for training
numTrain = round(trainRatio * numSamples);

% Split the data into training and testing sets
trainData = featureMatrix(idx(1:numTrain), :);
testData = featureMatrix(idx(numTrain+1:end), :);
trainLabels = numericLabels(idx(1:numTrain));
testLabels = numericLabels(idx(numTrain+1:end));



% Check for NaN values in numericLabels
nanIndices = isnan(numericLabels);
disp(['Number of NaN values in numericLabels: ', num2str(sum(nanIndices))]);


% Filter out NaN values from numericLabels and corresponding rows in trainData
validIndices = ~isnan(numericLabels);
numericLabels = numericLabels(validIndices);
trainData = trainData(validIndices, :);

% Check if there are still valid samples for training
if isempty(trainData) || isempty(numericLabels)
    error('No valid samples remaining after NaN removal.');
end

% Check if there are still valid classes after filtering NaN values
uniqueClasses = unique(numericLabels);
if isempty(uniqueClasses)
    error('No valid classes remaining after NaN removal.');
end

% Convert numericLabels to a categorical array with unique classes
classLabels = categorical(numericLabels, uniqueClasses);

% Create a template for an SVM classifier
t = templateSVM('KernelFunction', 'linear', 'Standardize', true);

% Train a multiclass SVM classifier
svmClassifier = fitcecoc(trainData, classLabels, 'Learners', t);

% Make predictions on the test data
predictions = predict(svmClassifier, testData);

% Evaluate the accuracy of the SVM
accuracy = sum(predictions == testLabels) / num



% Define a function to extract labels from filenames
function label = extractLabel(filename)
    % Split the filename based on the underscore
    parts = strsplit(filename, '_');
    % Extract the label from the second part (after the underscore)
    label = parts{2};
end
