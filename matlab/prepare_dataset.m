clear all, close all;
raw_folder_name = 'raw';
local_folder_path = './matlab/data/'; % needs to run from root folder
prepared_folder_name = 'prepared';

test_split = 0.3;
val_split = 0.1;
train_split = 1-test_split-val_split;

% Create directories for prepared data
type_names = {'test', 'train', 'validation'};
for type_idx = 1:length(type_names)
    mkdir(fullfile(local_folder_path, prepared_folder_name, type_names{type_idx}));
end

% Load raw data files from the specified folder
raw_files = dir(fullfile(local_folder_path, raw_folder_name, '*.csv'));