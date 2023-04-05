function load_and_plot_psychometric(fitpath, varargin)
parser = inputParser;
addParameter(parser, 'axes', [], ...
    @(x) validateattributes(x, {'matlab.graphics.axis.Axes'},{'scalar'}))
addParameter(parser, 'edges', [-inf, -30:10:30, inf], ...
    @(x) validateattributes(x, {'numeric'},{'vector'}))
addParameter(parser, 'legend', true, @(x) islogical(x) && isscalar(x))
addParameter(parser, 'observed_CI_facecolor', zeros(1,3))
addParameter(parser, 'observed_mean_linespec', '', @(x) ischar(x))
addParameter(parser, 'predicted_mean_linespec', 'k-', @(x) ischar(x))
addParameter(parser, 'resultsfolder', 'final', @(x) ischar(x))
parse(parser, varargin{:});
P = parser.Results; 
fitpath = char(fitpath);
load(fullfile(fitpath, [P.resultsfolder '\expectedemissions.mat']), ...
        'expectedemissions')
load(fullfile(fitpath, 'data.mat'), 'data')
choices = cellfun(@(x) x.choice, data{1}.trials);
Deltaclicks = cellfun(@(x) numel(x.clicktimes.R) - numel(x.clicktimes.L), data{1}.trials);
Echoices = cellfun(@(x) x.rightchoice, expectedemissions{1});
FHMDDM.plot_psychometric(choices, Deltaclicks, Echoices, varargin{:})
