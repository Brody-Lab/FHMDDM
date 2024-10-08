function plot_psychometric(choices, Deltaclicks, Echoices, varargin)
% PLOT_PSYCHOMETRIC plots the observed and predicted psychometric functions in the same axes
%   PLOT_PSYCHOMETRIC(`choices`, `Deltaclicks`, `Echoices`) takes the following inputs:
%
%       `choices`       a logical vector representing the behavioral choice on each trial (false; 
%                       left, true: right), the integer vector
%
%       `Deltaclicks`   an integer vector indicating the number of right minus left on each trial, 
%                       and the float vector
%
%       `Echoices`      a float vector representing the expected fraction of a right choice on each 
%                       trial
%
%   PLOT_PSYCHOMETRIC(...,PARAM1,VAL1,PARAM2,VAL2,...) specifies one or more of the following
%   name/value pairs:
%
%       `axes`                      The axes object in which the functions are drawn
%       `edges`                     A vector used for binning the difference between right and left
%                                   clicks
%       `legend`                    Whether a legend should be displayed
%       `observed_CI_facecolor`     A RGB triplet indicating the color of the shading representing
%                                   the 95% confidence interval
%       `observed_mean_linespec`    A char array specifying the display of the line representing the
%                                   observed means. If empty, no line is drawn,
%       `predicted_mean_linespec`   A char array specifying the display of the line representing the
%                                   predicted means. If empty, no line is drawn,
%
%   Example:
%       Deltaclicks = randi(80,500,1)-40;
%       Echoices = 1./(1+exp(-Deltaclicks/5));
%       choices = rand(500,1) <= Echoices;
%       FHMDDM.plot_psychometric(choices, Deltaclicks, Echoices)
validateattributes(choices, {'logical'},{'vector'})
validateattributes(Echoices, {'numeric'},{'vector'})
validateattributes(Deltaclicks, {'numeric'},{'integer', 'vector'})
parser = inputParser;
addParameter(parser, 'axes', [], ...
    @(x) validateattributes(x, {'matlab.graphics.axis.Axes'},{'scalar'}))
addParameter(parser, 'edges', [-inf, -30:10:30, inf], ...
    @(x) validateattributes(x, {'numeric'},{'vector'}))
addParameter(parser, 'legend', true, @(x) islogical(x) && isscalar(x))
addParameter(parser, 'observed_CI_facecolor', zeros(1,3))
addParameter(parser, 'observed_mean_linespec', '', @(x) ischar(x))
addParameter(parser, 'predicted_mean_linespec', 'k-', @(x) ischar(x))
addParameter(parser, 'resultsfolder', 'results', @(x) ischar(x))
parse(parser, varargin{:});
P = parser.Results; 
if isempty(P.axes)
    figure('pos', [100 100 300 325])
    set(gca, 'position', [0.15, 0.15, 0.8, 0.5])
else
    axes(P.axes)
end
FHMDDM.stylizeaxes
groupindices = discretize(Deltaclicks, [-inf, -30:10:30, inf]);
groupDeltaclicks = splitapply(@mean, Deltaclicks, groupindices);
[obsv, obsvci] = splitapply(@(x) binofit(sum(x), numel(x)), choices, groupindices);
pred = splitapply(@mean, Echoices, groupindices);
handles = [];
if ~all(P.observed_CI_facecolor == 1)
    handles = [handles; FHMDDM.shadeplot(groupDeltaclicks, obsvci(:,1), obsvci(:,2), ...
        'facecolor', P.observed_CI_facecolor)];
end
if ~isempty(P.observed_mean_linespec)
    plot(groupDeltaclicks, obsv, P.observed_mean_linespec);
end
if ~isempty(P.predicted_mean_linespec)
    handles = [handles; plot(groupDeltaclicks, pred, P.predicted_mean_linespec, 'linewidth', 1.5)];
end
if P.legend && ~isempty(P.predicted_mean_linespec)
    hlegend = legend(handles, {'obsv. 95%CI', 'predicted'}, 'location', 'best', 'AutoUpdate', 'off');
    set(hlegend, 'position', [0.42 0.72, 0.5, 0.15]);
end
absxlim = ceil(max(groupDeltaclicks)/10)*10;
xlim([-absxlim, absxlim])
xticks([-absxlim, 0, absxlim])
ylim([0,1])
yticks([0 1])
ylabel('fraction\newlinechose\newlineright', 'rotation', 0)
xlabel('#right - #left clicks')