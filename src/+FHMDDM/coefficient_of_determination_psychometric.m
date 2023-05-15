function R2 = coefficient_of_determination_psychometric(trials, resultspath, varargin)
% goodness-of-fit of the psychometric function
parser = inputParser;
addParameter(parser, 'edges', [-inf, -30:10:30, inf], ...
    @(x) validateattributes(x, {'numeric'},{'vector'}))
addParameter(parser, 'resultsfolder', 'results', @(x) ischar(x))
parse(parser, varargin{:});
P = parser.Results; 
load(fullfile(resultspath, 'expectedemissions'),'expectedemissions')
choices = cellfun(@(x) x.choice, trials);
Deltaclicks = cellfun(@(x) numel(x.clicktimes.R) - numel(x.clicktimes.L), trials);
Echoices = cellfun(@(x) x.rightchoice, expectedemissions{1});
groupindices = discretize(Deltaclicks, P.edges);
obsv = splitapply(@(x) binofit(sum(x), numel(x)), choices, groupindices);
pred = splitapply(@(x) binofit(sum(x), numel(x)), Echoices, groupindices);
SSresidual = sum(((pred - obsv)).^2);
SStotal = sum((mean(obsv) - obsv).^2);
R2 = 1 - SSresidual/SStotal;