clear all, close all

% data_folder_path = '/Users/jack/genSecSysId-Data/data/forced_pendulum/prepared/train';
data_folder_path = '/Users/jack/genSecSysId-Data/data/SilverboxFiles/prepared/train';
% Get a list of all CSV files in the specified folder
inputs_name = {'V1'};
output_name = {'V2'};

csv_files = dir(fullfile(data_folder_path, '*.csv'));
% Initialize a cell array to hold the input and output data
inputs_data = cell(length(csv_files), 1);
outputs_data = cell(length(csv_files), 1);

% Load each CSV file and store the specified input and output data
for i = 1:length(csv_files)
    file_path = fullfile(data_folder_path, csv_files(i).name);
    temp_data = readtable(file_path);
    
    % Extract the specified input and output columns
    inputs_data{i} = temp_data{:, inputs_name};
    outputs_data{i} = temp_data{:, output_name};
end

% Concatenate all input and output data into one
inputs = vertcat(inputs_data{:});
outputs = vertcat(outputs_data{:});

m_u = mean(inputs); std_u = std(inputs);
m_y = mean(outputs); std_y = std(outputs);

u_n = (inputs-m_u)/std_u;
y_n = (outputs-m_y)/std_y;

% sampling time
f_max = 200;
odd_harmonics = 1342;
l_max = 2*odd_harmonics - 1;
% f_max = l_max * f_0
f0 = f_max/l_max;
fs = f0 * 8192;
Ts = 1/fs;

% Run n4sid to identify a linear state-space model
tt1 = iddata(y_n, u_n, Ts);
opt = n4sidOptions('Focus', 'simulation', 'EnforceStability', true);
nx = 10;
sys = n4sid(tt1, nx, opt); 
compare(tt1,sys)

A = sys.A; B=sys.B; C = sys.C; D = sys.D;B
save('n4sid_params.mat', 'A', 'B', 'C','D')



%% whole dataset
% % load mat file
% mat_filename = '/Users/jack/genSecSysId-Data/data/SilverboxFiles/SNLS80mV.mat';
% d = load(mat_filename);
% 
% u_full = d.V1; y_full = d.V2;
% 
% m_u = mean(u_full); std_u=std(u_full);
% m_y = mean(y_full); std_y=std(y_full);
% % normalize
% u_full_n = (u_full - m_u) / std_u;
% y_full_n = (y_full - m_y) / std_y;
% 
% 
% tt1 = iddata(y_full_n', u_full_n', Ts);
% opt = n4sidOptions('Focus', 'simulation', 'EnforceStability', true);
% nx = 10;
% sys = n4sid(tt1,nx,opt); 
% compare(tt1,sys)





