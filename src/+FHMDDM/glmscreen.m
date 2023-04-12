function DeltaLL = glmscreen(choices, spiketimes_s, trialstart_s, varargin)
validateattributes(choices, {'logical'}, {'vector'})
validateattributes(spiketimes_s, {'numeric'}, {'vector'})
validateattributes(trialstart_s, {'numeric'}, {'vector'})
assert(numel(choices)==numel(trialstart_s))
P = inputParser;
addParameter(P, 'nbasisfunctions', 8, ...
    @(x) validateattributes(x, {'numeric'}, {'nonnegative'}))
addParameter(P, 'nsimulations', 100, ...
    @(x) validateattributes(x, {'numeric'}, {'positive', 'integer', 'scalar'}))
addParameter(P, 'duration_s', 1, ...
    @(x) validateattributes(x, {'numeric'}, {'scalar'}))
parse(P, varargin{:});
P = P.Results;
ntrials = numel(trialstart_s);
spiketimes_s= spiketimes_s - min(trialstart_s);
trialstart_s = trialstart_s - min(trialstart_s);
y = nan(ntrials,1);
for i = 1:ntrials
    y(i) = sum(spiketimes_s >= trialstart_s(i) & ...
               spiketimes_s < trialstart_s(i) + P.duration_s)/P.duration_s;
end
trialstart_timestep = floor(trialstart_s/P.duration_s)+1;
ntimesteps = ceil(max(trialstart_s)/P.duration_s)*P.duration_s;
%%
if P.nbasisfunctions == 1
    X = [ones(ntrials,1), choices];
else
    F = NeuroGLMCovariate.raisedcosines(P.nbasisfunctions,0,ntimesteps, 'stretch', 1e-6);
    X = [F(trialstart_timestep,:), choices];
end
w = glmfit(X, y, 'poisson', 'constant', 'off');
lambda = exp(X*w);
meanLL = mean(log(poisspdf(y, lambda)));
meanLL0 = 0;
for i = 1:P.nsimulations
    meanLL0 = meanLL0 + mean(log(poisspdf(poissrnd(lambda),lambda)))/P.nsimulations;
end
DeltaLL = meanLL-meanLL0;