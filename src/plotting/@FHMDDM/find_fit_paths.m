function fit_paths = find_fit_paths(analysisname)
    top_level_folder = fullfile(FHMDDM().analysis_data_folder_path_windows, analysisname);
    stuff_in_top_level_folder = dir(top_level_folder);
    fit_paths = stuff_in_top_level_folder([stuff_in_top_level_folder.isdir]);
    fit_paths = arrayfun(@(x) string(fullfile(x.folder, x.name)), fit_paths(3:end));
end