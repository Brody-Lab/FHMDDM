function makereport(modelname, outputpath, analysisfolder, varargin)
% generate a report of an analysis
%
% ARGUMENT
%
%   modelname
%       name of the model
%
%   outputpath
%       path of the model outputs
%
%   reportpath
%       path of the report and the figures of the report
validateattributes('datapath', {'char'}, {'row'})
validateattributes('modelname', {'char'}, {'row'})
validateattributes('outputpath', {'char'}, {'row'})
validateattributes('reportfolder', {'char'}, {'row'})
parser = inputParser;
addParameter(parser, 'resultsfolder', 'results', @(x) ischar(x))
parse(parser, varargin{:});
P = parser.Results; 
reportfolder = fullfile(analysisfolder, ['report_' modelname]);
if ~isfolder(reportfolder)
    mkdir(reportfolder)
end
[~, topfoldername] = fileparts(fileparts(analysisfolder));
[~, analysisname] = fileparts(analysisfolder);
markdownpath_prefix = ['\n<img src="/' topfoldername '/' analysisname '/' ['report_' modelname]];
%% psychometric
close all
load(fullfile(outputpath, [P.resultsfolder '\expectedemissions.mat']), 'expectedemissions')
load(fullfile(outputpath, 'trialsets.mat'), 'trialsets')
choices = cellfun(@(x) x.choice, trialsets{1}.trials);
Deltaclicks = cellfun(@(x) numel(x.clicktimes.R) - numel(x.clicktimes.L), trialsets{1}.trials);
Echoices = cellfun(@(x) x.rightchoice, expectedemissions{1});
FHMDDM.plot_psychometric(choices, Deltaclicks, Echoices)
psychometricpath =  fullfile(reportfolder, 'psychometric.svg');
saveas(gcf, psychometricpath)
%% R2
close all
figure('pos', [1e2, 1e2, 325, 225])
FHMDDM.prepareaxes
R2 = FHMDDM.coefficient_of_determination_one_session(outputpath, 'resultsfolder', P.resultsfolder);
h = histogram(max(R2,0), 0:0.1:1);
plot(median(R2), max(ylim), 'kv', 'markerfacecolor', 'k', 'markersize', 10)
xlim(0:1)
xticks(xlim)
set(h, 'facecolor', 'k', 'facealpha', 0.2)
ylabel('neurons')
xlabel('coefficient of determination (R^2)')
saveas(gcf,  fullfile(reportfolder, 'R2.svg'));
%% PSTH
PSTH = load(fullfile(outputpath, [P.resultsfolder  '\pethsets_stereoclick.mat']));
nneurons = size(PSTH.pethsets{1},1);
time_s = PSTH.time_s;
time_s = time_s(time_s<=1);
figure('pos', [100 100 600 300])
for n = 1:nneurons
    clf
    set(gca, 'position', [0.1, .2, 0.4, 0.7])
    for condition = ["leftchoice", "rightchoice"]
        FHMDDM.plot_peth(PSTH.pethsets{1}{n}, condition, time_s, ...
            'linestyle_observed_mean', '')
    end
    handles = get(gca, 'Children');
    hlegend = legend(handles([5 2 4 1]), {'observed 95%CI, right choice', ...
        'observed 95%CI, left choice', 'prediction, right choice', ...
        'prediction, left choice'});
    set(hlegend, 'position', [0.575 0.5, 0.35, 0.2]);
    saveas(gcf, [reportfolder, filesep 'psth_conditioned_on_choice_neuron_' num2str(n, '%02i') '.svg'])
    clf
    set(gca, 'position', [0.1, .2, 0.4, 0.7])
    for condition = "unconditioned"
        FHMDDM.plot_peth(PSTH.pethsets{1}{n}, condition, time_s, ...
            'linestyle_observed_mean', '')
    end
    handles = get(gca, 'Children');
    hlegend = legend(handles([2 1]), {'observed 95%CI', 'prediction'});
    set(hlegend, 'position', [0.575 0.5, 0.35, 0.15]);
    saveas(gcf, [reportfolder, filesep 'psth_unconditioned_neuron_' num2str(n, '%02i') '.svg'])
end
%% generate a text file
fileID = fopen([reportfolder '/report_' modelname '.md'], 'w');
fprintf(fileID, '\n# psychometric\n');
fprintf(fileID, [markdownpath_prefix '/psychometric.svg" height="300">\n']);
fprintf(fileID, '\n# PSTH goodness-of-fit\n');
fprintf(fileID, [markdownpath_prefix '/R2.svg" height="200">\n']);
for n = 1:nneurons
    fprintf(fileID, '\n## neuron %02i\n', n);
    fprintf(fileID, [markdownpath_prefix '/psth_conditioned_on_choice_neuron_' num2str(n, '%02i') ...
        '.svg" height="300">\n']);
    fprintf(fileID, [markdownpath_prefix '/psth_unconditioned_neuron_' num2str(n, '%02i') ...
        '.svg" height="300">\n']);
end
fclose(fileID);