function S = pericommitment_spiketrains(paccumulator, trials, varargin)
% Spike train of each around the inferred time of decision commitment
%
% ARGUMENT
%-`paccumulator`: moment-to-moment probability of the accumulator. Element `paccumulator{m}{t}(i)`
%corresponds to the i-th state of the accumulator at the t-th time step of the m-th trial. The first
%and last state of accumulator are assumed to correspond to commitment to the left and right choice,
%respectively.
%-`trials`: a cell array whose each element is a structure containing the moment-to-moment spike
%count of each neuron. The element `trials{m}.spiketrains{n}(t)` corresponds to the spike train
%response of the n-th neuron at the t-th time step on the m-th trial.
%
% OUTPUT
%-`S`: a structure with the following fields
%   `timesteps`: time relative to the inferred moment of commitment
%   `left` (`right`): a cell vector whose each element is a `nneurons`-by-`ntimesteps` matrix
%   indicating the response aligned to being close to the `left` (`right`) bound
%
% OPTIONAL ARGUMENT
%-`commitmentthreshold`: probability of the accumulator considered for a commitment
%-timesteps: time relative to the inferred moment of commitment
%-`randomize`: as a negative control, for each group of trials associated with either
% the left or the right choice, randomize the inferred time of commitment
parser = inputParser;
addParameter(parser, 'commitmentthreshold', 0.8, @(x) isnumeric(x) && isscalar(x))
addParameter(parser, 'randomize', false, @(x) isscalar(x) && islogical(x))
addParameter(parser, 'nsteps_before_commitment', 30, @(x) ...
    validateattributes(x, {'numeric'}, {'scalar', 'positive', 'integer'}))
addParameter(parser, 'nsteps_after_commitment', 30, @(x) ...
    validateattributes(x, {'numeric'}, {'scalar', 'positive', 'integer'}))
addParameter(parser, 'filtersigma_timesteps', 10, @(x) isscalar(x) && isnumeric(x))
addParameter(parser, 'filterwidth_timesteps', 30, @(x) isscalar(x) && isnumeric(x))
parse(parser, varargin{:});
P = parser.Results;
ntrials = numel(paccumulator);
[t_commitleft, t_commitright] = deal(nan(ntrials,1));
for m = 1:ntrials
    commitleft = cellfun(@(x) x(1) > P.commitmentthreshold, paccumulator{m});
    commitright = cellfun(@(x) x(end) > P.commitmentthreshold, paccumulator{m});
    t = find(commitleft, 1);
    if ~isempty(t)
         t_commitleft(m) = t;
    end
    t = find(commitright, 1);
    if ~isempty(t)
         t_commitright(m) = t;
    end
end
Commit = struct;
Commit.left.trial = find(~isnan(t_commitleft));
Commit.left.timestep = t_commitleft(~isnan(t_commitleft));
Commit.right.trial = find(~isnan(t_commitright));
Commit.right.timestep = t_commitright(~isnan(t_commitright));
if P.randomize
    for side = ["left", "right"]
        ntrials = numel(Commit.(side).timestep);
        randomindices = randperm(ntrials);
        Commit.(side).timestep = Commit.(side).timestep(randomindices);
        for m = 1:ntrials
            trialindex = Commit.(side).trial(m);
            Commit.(side).timestep(m) = min(trials{trialindex}.ntimesteps,Commit.(side).timestep(m));
        end
    end
end
nneurons = numel(trials{1}.spiketrains);
S = struct;
S.timesteps = -P.nsteps_before_commitment:P.nsteps_after_commitment;
npre = P.nsteps_before_commitment;
npost = P.nsteps_after_commitment;
for side = ["left", "right"]
    S.(side) = cell(nneurons,1);
    [S.(side){:}] = deal(nan(numel(Commit.(side).trial), numel(S.timesteps)));
    for m = 1:numel(Commit.(side).trial)
        trialindex = Commit.(side).trial(m);
        pre = min(npre, Commit.(side).timestep(m)-1);
        post = min(npost, trials{trialindex}.ntimesteps-Commit.(side).timestep(m));
        timesteps = Commit.(side).timestep(m)+(-pre:post);
        timesteps_dc = npre+1+(-pre:post);        
        for n = 1:nneurons
            y = trials{trialindex}.spiketrains{n}(timesteps);
            if P.filtersigma_timesteps > 0
                y = M23a.convolve_counts_causalgaussian(y, P.filterwidth_timesteps, ...
                    P.filtersigma_timesteps);
            end
            S.(side){n}(m,timesteps_dc) = y;
        end
    end
end