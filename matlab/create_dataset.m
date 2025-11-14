clear all, close all;
generation_type = {'rand-input_noise', 'rand-input_sin', 'zero-input_noise','zero-input_sin'};
local_folder_path = './data/'; % needs to run from root folder
raw_folder_name = 'raw';
raw_folder_path = fullfile(local_folder_path,raw_folder_name);
if ~exist(raw_folder_path, 'dir')
    mkdir(raw_folder_path);
end

for i = 1:length(generation_type)
    gen_type = generation_type{i};
    data_filename = sprintf('init_%s.mat' , gen_type);
    load(fullfile(local_folder_path, data_filename))

    % some general data
    dt = dsys_.dt; nx = size(dsys_.Ad,1); nd = size(dsys_.Bd,2); ne = size(dsys_.Cd,1);
    nw = size(dsys_.C2d,2); nz = nw;
    N = size(feasible_ic_and_inputs{1}.d,2);
    t = linspace(0,(N-1)*dt, N);

    for ii=1:length(feasible_ic_and_inputs)
        data = feasible_ic_and_inputs{ii};
        tab = table(t', data.d', data.e', data.x(:,1:N)', ...
            'VariableNames', {'time', 'd', 'e', 'x'});
            % Save the table to a CSV file
        csv_filename = sprintf('init_%s-output_%i.csv', gen_type, ii);
        fullfilename = fullfile(raw_folder_path, csv_filename);
        writetable(tab, fullfilename);
        fprintf('%i/%i: write %s to %s\n', ii, length(feasible_ic_and_inputs), gen_type, fullfilename)
    end
    
end