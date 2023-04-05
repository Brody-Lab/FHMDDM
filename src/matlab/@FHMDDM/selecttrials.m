function trialindices = selecttrials(Trials, configurationname)
% Select trials according to named rules
%
% ARGUMENT
%-`Trials`: a structure containing the behavioral data
%
% OUTPUT
%-`trialindices`: a vector of logicals representing which trial is to be included
validateattributes(configurationname, {'char', 'string'},{})
spreadsheetpath = [fileparts(mfilename('fullpath')), filesep ...
    'configurations'  filesep, 'trialselection.csv'];
T = readtable(spreadsheetpath);
T.name = string(T.name);
T.trial_type = string(T.trial_type);
assert(size(T,1) == numel(unique(T.name)), 'Redundant selection rules')
index = T.name == configurationname;
assert(sum(index)==1, sprintf('Unknown rule named %s', configurationname))
trialindices = true(numel(Trials.pokedR),1);
if T.trial_type(index) ~= ""
    trialindices = trialindices & Trials.trial_type == T.trial_type{index};
end
if ~isnan(T.responded(index))
    trialindices = trialindices & Trials.responded == T.responded(index);
end
if ~isnan(T.violated(index))
    trialindices = trialindices & Trials.violated == T.violated(index);
end
if ~isnan(T.laseron(index))
    trialindices = trialindices & Trials.laser.isOn == T.laseron(index);
end
if sum(trialindices) > T.maxtrials(index)
    trialindices = rand_select(trialindices, T.maxtrials(index));
end