function copySubset(files, indices, destination)
    for i = indices
        src = fullfile(files(i).folder, files(i).name);
        dst = fullfile(destination, files(i).name);
        copyfile(src, dst);
    end
end