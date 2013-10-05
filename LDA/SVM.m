function [ accuracy ] = SVM( training_data, testing_data )
%SVM Train an SVM on the training data and validate it on the testing data.
%   Inputs: training_data: an N x D matrix of data. Contains labels in the
%           final column.
%           testing_data: an N x D matrix of data. Contains labels in the
%           final column, which are used for producing the F-measure.
%   Output: accuracy: The classification's accuracy.

% We will use MATLAB's SVM functions to perform classification.
% The first step is to train the SVM. 
classifier = svmtrain(training_data(:, 1:end-1), training_data(:, end));

% The second step is to classify the test data.
predictedLabs = svmclassify(classifier, testing_data(:, 1:end - 1));

% Given both the true and predicted labels, we can now calculate the 
% F-measure.

trueLabs = testing_data(:, end);
assert(length(trueLabs) == length(predictedLabs));
truePositives = length(predictedLabs(predictedLabs == trueLabs));
accuracy = truePositives / length(trueLabs);

end

