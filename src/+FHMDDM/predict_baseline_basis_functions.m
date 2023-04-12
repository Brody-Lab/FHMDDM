function S = predict_baseline_basis_functions(spiketimes_s, trialstart_s, timestep_s, varargin)
validateattributes(spiketimes_s, {'numeric'}, {'vector'})
validateattributes(trialstart_s, {'numeric'}, {'vector'})
P = inputParser;
addParameter(P, 'kfold', 10, ...
    @(x) validateattributes(x, {'numeric'}, {'integer', 'scalar', 'positive'}))
addParameter(P, 'nbasisfunctions', 1:8, ...
    @(x) validateattributes(x, {'numeric'}, {'nonnegative'}))
parse(P, varargin{:});
P = P.Results;
ntrials = numel(trialstart_s);
spiketimes_s= spiketimes_s - min(trialstart_s);
trialstart_s = trialstart_s - min(trialstart_s);
tbeg = min(trialstart_s);
tend = ceil(max(trialstart_s)/timestep_s)*timestep_s;
binedges_s = tbeg:timestep_s:tend;
spikecounts = histcounts(spiketimes_s, binedges_s)';
trialstart_timestep = floor(trialstart_s/timestep_s)+1;
trialtimesteps = trialstart_timestep;
y = spikecounts(trialtimesteps);
ntimesteps = tend;
trialtimesteps = floor(trialstart_s/timestep_s)+1;
x0 = ones(ntimesteps,1);
%%
[LL, LLnorm] = deal(zeros(numel(P.nbasisfunctions),1));
for i = 1:numel(P.nbasisfunctions)
    nbf = P.nbasisfunctions(i);
    if nbf == 1
        X = x0;
    else
        F = NeuroGLMCovariate.raisedcosines(nbf,0,ntimesteps, 'stretch', 1e-6);
        [U,~,~] = svd(F);
        X = U(trialtimesteps,1:nbf);
    end
    cvp = cvpartition(ntrials, 'KFold', P.kfold);
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
LL = LL./ntrials;
LLnorm = LLnorm./ntrials;
%%
[~, bestindex] = max(LL);
if bestindex == 1
    baselines_hz = [];
    best_nbasisfunctions = 1;
else
    best_nbasisfunctions = P.nbasisfunctions(bestindex);
    F = NeuroGLMCovariate.raisedcosines(best_nbasisfunctions,0,ntimesteps, 'stretch', 1e-6);
%     [U,~,~] = svd(F);
%     X = U(trialtimesteps,1:best_nbasisfunctions);
    X = F(trialtimesteps,1:best_nbasisfunctions);
    w = glmfit(X, y, 'poisson', 'constant', 'off');
    baselines_hz = exp(X*w);
end
%%
S = struct;
S.best_nbasisfunctions = best_nbasisfunctions;
S.baselines_hz = baselines_hz;
S.LL = LL;
S.LLnorm = LLnorm;
S.nbasisfunctions = P.nbasisfunctions;
S.kfold = P.kfold;
S.timestep_s = timestep_s;
S.trialtimesteps = trialtimesteps;
S.y = y;