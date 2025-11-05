clear all, close all;
raw_folder_name = 'raw';
local_folder_path = './matlab/data/'; % needs to run from root folder
prepared_folder_name = 'prepared';
prepared_path = fullfile(local_folder_path, prepared_folder_name);
if ~exist(prepared_path, 'dir')
    mkdir(prepared_path);
end

test_split = 0.3;
val_split = 0.1;
train_split = 1-test_split-val_split;
splits = {test_split, train_split, val_split};

% Load raw data files from the specified folder
files = dir(fullfile(local_folder_path, raw_folder_name, '*.csv'));
nFiles = numel(files);
if nFiles == 0
    error('No CSV files found in folder: %s', sourceFolder);
end

% Random permutation of indices
idx = randperm(nFiles);

% Compute split indices
nTest = round(test_split * nFiles);
nVal = round(val_split * nFiles);
nTrain = nFiles - nTest - nVal;

trainIdx = idx(1:nTrain);
valIdx   = idx(nTrain+1:nTrain+nVal);
testIdx  = idx(nTrain+nVal+1:end);

% Create subfolders if needed
trainFolder = fullfile(prepared_path, 'train');
valFolder   = fullfile(prepared_path, 'validation');
testFolder  = fullfile(prepared_path, 'test');
mkdir(trainFolder);
mkdir(valFolder);
mkdir(testFolder);

% Helper function to copy files
copySubset(files, trainIdx, trainFolder);
copySubset(files, valIdx, valFolder);
copySubset(files, testIdx, testFolder);

fprintf('Split %d files into:\n', nFiles);
fprintf('  Train: %d\n', nTrain);
fprintf('  Val:   %d\n', nVal);
fprintf('  Test:  %d\n', nTest);





