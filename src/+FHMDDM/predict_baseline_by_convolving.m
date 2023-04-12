function S = predict_baseline_by_convolving(spiketimes_s, trialstart_s, trialend_s, varargin)
% predict the baseline firing rate of each trial by convolving the sequence of spike counts
%
% ARGUMENT
%-`spiketimes_s`: a vector of floats containing the spike times, in a seconds, of a neuron
%-`trialstart_s`: a vector of floats containing the start time, in seconds, of each trial
%-`trialend_s`: a vector of floats containing the end time, in seconds, of each trial
%
% RETURN
%-`S`: a structure containing the following fields:
%   --`baselines_hz`: a vector of floats representing the estimated within-trial baseline firing rate.
%   --`bestsigma_s`: the standard deviation of the symmetric gaussian filter that yielded the best 
%   --
%
% OPTIONAL ARGUMENT
%-`recordingstart_s`: starting time of the recording; the defaul 
%-`recordingend_s`: termination time of the recording; the default is assumed to be the last spike
% of the neuron recorded
%-`sigmas`: a vector of floats containing the standard deviations of the symmetric gaussian filter,
%in terms of time steps
%-`timestep_s`: duration, in seconds, of each time step
validateattributes(spiketimes_s, {'numeric'}, {'vector'})
validateattributes(trialstart_s, {'numeric'}, {'vector'})
validateattributes(trialend_s, {'numeric'}, {'vector'})
assert(numel(trialstart_s) == numel(trialend_s));
assert(all(trialend_s>trialstart_s))
P = inputParser;
addParameter(P, 'kfold', 10, ...
    @(x) validateattributes(x, {'numeric'}, {'integer', 'scalar', 'positive'}))
addParameter(P, 'recordingstart_s', min(spiketimes_s), ...
    @(x) validateattributes(x, {'numeric'}, {'scalar', 'positive'}))
addParameter(P, 'recordingend_s', max(spiketimes_s), ...
    @(x) validateattributes(x, {'numeric'}, {'scalar', 'positive'}))
addParameter(P, 'sigmas', [0.1, round(10.^(0:0.2:3))], ...
    @(x) validateattributes(x, {'numeric'}, {'nonnegative'}))
addParameter(P, 'timestep_s', 1, ...
    @(x) validateattributes(x, {'numeric'}, {'scalar', 'nonnegative'}))
parse(P, varargin{:});
P = P.Results;
assert(min(trialstart_s) > P.recordingstart_s);
assert(max(trialend_s) < P.recordingend_s);
assert(min(spiketimes_s) >= P.recordingstart_s);
assert(max(spiketimes_s) <= P.recordingend_s);
recordingstart_s = P.recordingstart_s - min(trialstart_s);
recordingend_s = P.recordingend_s - min(trialstart_s);
spiketimes_s = spiketimes_s - min(trialstart_s);
trialend_s = trialend_s - min(trialstart_s);
trialstart_s = trialstart_s - min(trialstart_s);
tbeg = ceil(recordingstart_s/P.timestep_s)*P.timestep_s;
tend = floor(recordingend_s/P.timestep_s)*P.timestep_s;
trialstart_timestep = floor(trialstart_s/P.timestep_s)+1 + abs(tbeg);
trialstend_timestep = ceil(trialend_s/P.timestep_s)+1+ abs(tbeg);
ntrials = numel(trialstart_timestep);
%%
binedges_s = tbeg:P.timestep_s:tend;
spikecounts = histcounts(spiketimes_s, binedges_s)';
intrialindices = cell(ntrials,1);
for i = 1:ntrials
    intrialindices{i} = (trialstart_timestep(i):trialstend_timestep(i))';
end
intrialindices = cell2mat(intrialindices);
maskedspikecounts = spikecounts;
maskedspikecounts(intrialindices) = NaN;
trialtimesteps = trialstart_timestep;
y = spikecounts(trialtimesteps);
x0 = ones(numel(trialtimesteps),1);
%%
[LL, LLnorm] = deal(zeros(numel(P.sigmas),1));
for i = 1:numel(P.sigmas)
    if P.sigmas(i) < 1
        X = x0;
    else
        x = gaussianfiltercounts(maskedspikecounts, 'type', 'symmetric', ...
            'sigma', P.sigmas(i));
        x = x(trialtimesteps);
        x(isnan(x)) = 0;
        x = x - mean(x);
        X = [x0, x];
    end
    cvp = cvpartition(ntrials, 'kfold', P.kfold);
    for k = 1:P.kfold
        Xtrain = X(cvp.training(k),:);
        ytrain = y(cvp.training(k));
        w = glmfit(Xtrain, ytrain, 'poisson', 'constant', 'off');
        Xtest = X(cvp.test(k),:);
        ytest = y(cvp.test(k));
        lambda = exp(Xtest*w);
        LL_k = log(poisspdf(ytest, lambda));            
        LL(i) = LL(i) + sum(LL_k);
        normalization = log(2*pi*lambda)/2;
        normalization = max(normalization, 1);
        LLnorm(i) = LLnorm(i) + sum(LL_k ./ normalization);            
    end
end

LL = LL./numel(trialtimesteps);
LLnorm = LLnorm./numel(trialtimesteps);
%%
[~, bestindex] = max(LL);
if bestindex == 1
    baselines_hz = [];
    bestsigma_s = NaN;
else
    x = gaussianfiltercounts(maskedspikecounts, 'type', 'symmetric', ...
            'sigma', P.sigmas(bestindex));
    baselines_hz = x(trialtimesteps);
    bestsigma_s = P.sigmas(bestindex)*P.timestep_s;
end
%%
S = struct;
S.bestsigma_s = bestsigma_s;
S.baselines_hz = baselines_hz;
S.LL = LL;
S.LLnorm = LLnorm;
S.sigmas = P.sigmas;
S.kfold = P.kfold;
S.timestep_s = P.timestep_s;
S.trialtimesteps = trialtimesteps;
S.y = y;