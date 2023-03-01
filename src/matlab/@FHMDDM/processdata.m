function [] = processdata(configurationname)
% Process and save the data
%
%=ARGUMENT
%
%   configurationname
%       name of the configuration specifying the parameters of data processing
validateattributes(configurationname, {'char', 'string'},{})
spreadsheetpath = [fileparts(mfilename('fullpath')), filesep ...
        'configurations'  filesep, 'dataprocessing.csv'];
T = readtable(spreadsheetpath);
T.configurationname = string(T.configurationname);
T.neuronselection = string(T.neuronselection);
T.trialselection = string(T.trialselection);
index = find(T.configurationname == string(configurationname));
if numel(index) < 1
    error('cannot find configuration with the name "%s"', configurationname)
elseif numel(index) > 1
    error('multiple configurations with the name "%s"', configurationname)
end
T = T(T.configurationname == configurationname,:);
recordingsesions = readtable([fileparts(mfilename('fullpath')), filesep ...
        'logs'  filesep, 'recordingsessions.csv']);
folderpath = fullfile(T.folderpath{1}, T.configurationname{1});
if ~isfolder(folderpath)
    success = mkdir(folderpath);
    if ~success
        error('Failed to create folder "%s"', folderpath)
    end
end
for i = 1:size(recordingsesions,1)
    clear Cells Trials
    load(recordingsesions.Cellspath{i}, 'Cells')
    load(recordingsesions.Trialspath{i}, 'Trials')
    trialset = FHMDDM.maketrialset(Cells, Trials, configuration);
    filepath = [folderpath, filesep, recordingsesions.recording_id{i}, filesep, 'trialset.mat'];
    save(filepath, 'trialset', '-struct')
end

keyboard