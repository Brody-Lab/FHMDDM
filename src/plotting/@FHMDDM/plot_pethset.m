function plot_pethset(pethset, time_s, varargin)
% PLOT_PETHSET plot the peri-evidence time histogram (PETH) of a neuron in each task condition
%   PLOT_PETH(PETHSET, TIME_s) plots the PETH in each condition, specified by each field of the
%   struct PETHSET. The portion of the PETH to be plotted is specified by the vector TIME_S and by
%   the shortest PETH across task conditions. The nine task conditions are organized into four
%   subplots. The first subplot shows the PETH computed from all trials. The second shows the two
%   PETHs conditioned on sign of the evidence. The third the two PETHs conditioned on the
%   behavioral choice. And the fourth the four PETHs conditioned on the choice, the sign of the
%   evidence, and the strength of the evidence.
%
%   PLOT_PETH(...,PARAM1,VAL1,PARAM2,VAL2,...) specifies one or more of
%   the following name/value pairs:
%
%       `Figure`                    The figure in which the plots are made
%
%       'Linestyle_observed_mean'   A character row vector specifying the style of the line 
%                                   representing the observed mean. If it is empty, the mean is
%                                   not shown.
%
%       'Linestyle_predicted_mean'  A character row vector specifying the style of the line
%                                   representing the predicted mean. If empty, the mean is not
%                                   shown.
%
%       'Referenceevent'            A character row vector or a string specifying the event in the 
%                                   trial to which the PETH is aligned in time
%
%   Example:
%       % plot simulated peri-event time histograms
%       S = struct;
%       time_s = (0:99)*0.01;
%       Multipler = struct;
%       Multiplier.unconditioned = 2;
%       Multiplier.rightchoice = 4;
%       Multiplier.leftchoice = 1;      
%       Multiplier.rightevidence = 3;
%       Multiplier.leftevidence = 2;
%       Multiplier.rightchoice_strong_rightevidence = 5;
%       Multiplier.rightchoice_weak_rightevidence = 3;
%       Multiplier.leftchoice_strong_leftevidence = 0.5;
%       Multiplier.leftchoice_weak_leftevidence = 1.5;
%       for condition = string(fieldnames(Multiplier)')
%           S.(condition).predicted = sin(time_s)*Multiplier.(condition);
%           noisy = poissrnd(repmat(S.(condition).predicted,100,1));
%           S.(condition).observed = mean(noisy);
%           ci = bootci(1e3, @mean, noisy);
%           S.(condition).lowerconfidencelimit = ci(1,:);
%           S.(condition).upperconfidencelimit = ci(2,:);
%       end
%       figure('pos', [50, 50, 1600, 300])
%       FHMDDM.plot_pethset(S, time_s)
validateattributes(pethset, {'struct'},{'scalar'})
validateattributes(time_s, {'numeric'}, {})
parser = inputParser;
addParameter(parser, 'figure', [], @(x) validateattributes(x, {'matlab.ui.Figure'},{'scalar'}))
addParameter(parser, 'linestyle_observed_mean', '--', @(x) ischar(x))
addParameter(parser, 'linestyle_predicted_mean', '-', @(x) ischar(x))
addParameter(parser, 'referenceevent', 'stereoclick', @(x) validateattributes(x, {'char'},{'row'}))
parse(parser, varargin{:});
P = parser.Results; 
if isempty(P.figure)
    gcf;
else
    figure(P.figure)
end
nbins = length(time_s);
for condition = string(fieldnames(pethset)')
    for stat = string(fieldnames(pethset.(condition))')
        nbins = min(nbins, length(pethset.(condition).(stat)));
    end
end
time_s = time_s(1:nbins);
naxeses = 4;
axeses = nan(4,1);
k = 0;
k = k + 1;
axeses(k) = subplot(1,naxeses,k);
FHMDDM.plot_peth(pethset, "unconditioned", time_s, ...
        'linestyle_observed_mean', '', ...
        'linestyle_predicted_mean', P.linestyle_predicted_mean, ...
        'referenceevent', P.referenceevent, ...
        'show_observed_CI', true)
title('all trials')
k = k + 1;
axeses(k) = subplot(1,naxeses,k);
for condition = ["leftevidence", "rightevidence"]
    FHMDDM.plot_peth(pethset, condition, time_s, ...
        'linestyle_observed_mean', '', ...
        'linestyle_predicted_mean', P.linestyle_predicted_mean, ...
        'referenceevent', P.referenceevent, ...
        'show_observed_CI', true)
    title('grouped by choice')
end
k = k + 1;
axeses(k) = subplot(1,naxeses,k);
for condition = ["leftchoice", "rightchoice"]
    FHMDDM.plot_peth(pethset, condition, time_s, ...
        'linestyle_observed_mean', '', ...
        'linestyle_predicted_mean', P.linestyle_predicted_mean, ...
        'referenceevent', P.referenceevent, ...
        'show_observed_CI', true)
    title('grouped by evidence')
end
k = k + 1;
axeses(k) = subplot(1,naxeses,k);
for condition = ["leftchoice_weak_leftevidence", ...
                 "leftchoice_strong_leftevidence", ...
                 "rightchoice_weak_rightevidence", ...
                 "rightchoice_strong_rightevidence"]
    FHMDDM.plot_peth(pethset, condition, time_s, ...
        'linestyle_observed_mean', P.linestyle_observed_mean, ...
        'linestyle_predicted_mean', P.linestyle_predicted_mean, ...
        'referenceevent', P.referenceevent, ...
        'show_observed_CI', false)
    title('grouped by choice and evidence')
end
linkaxes(axeses, 'xy')