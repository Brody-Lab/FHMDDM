function timestep_commit = commitmenttimes(paccumulator, trials, varargin)
%{ 
Infer the time of decision commitment on each trial

ARGUMENT

    paccumulator
        moment-to-moment probability of the accumulator. Element `paccumulator{m}{t}(i)` corresponds
        to the i-th state of the accumulator at the t-th time step of the m-th trial. The first and
        last state of accumulator are assumed to correspond to commitment to the left and right
        choice, respectively.

    trials
        a cell array whose each element is a structure containing the moment-to-moment spike count
        of each neuron. The element `trials{m}.spiketrains{n}(t)` corresponds to the spike train
        response of the n-th neuron at the t-th time step on the m-th trial.

OUTPUT

    timestep_commit
        time step of commitment on each trial aligned to the stereoclick. A element with NaN value
        indicates that probability of the accumulator variable did not reach the threshold on that
        trial, for either the left or the right bound.

OPTIONAL NAME-VALUE PAIR ARGUMENT

    commitmentthreshold
        probability of the accumulator considered for a commitment. A scalar between 0 and 1
%}


% Spike train of each around the inferred time of decision commitment. 
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
%
%   timestep_commit
%       time of commitment on each trial, in seconds
%
% OPTIONAL ARGUMENT
%
%   commitmentthreshold
%       probability of the accumulator considered for a commitment
validateattributes(paccumulator, {'cell'}, {'vector'})
validateattributes(trials, {'cell'}, {'vector'})
parser = inputParser;
addParameter(parser, 'threshold', 0.8, @(x) isnumeric(x) && isscalar(x) && (x>0) && (x<1))
parse(parser, varargin{:});
P = parser.Results;
ntrials = numel(paccumulator);
timestep_commit = nan(ntrials,1);
for i = 1:ntrials
    commitleft = cellfun(@(x) x(1) > P.threshold, paccumulator{i});
    commitright = cellfun(@(x) x(end) > P.threshold, paccumulator{i});
    t = min([find(commitleft, 1); find(commitright, 1)]);
    if ~isempty(t)
        timestep_commit(i) = t;
    end
end