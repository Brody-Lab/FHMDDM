function tuningcurve = tuning_curve_accumulated_evidence(pa, spiketimes_s, starttimes_s, varargin)
%{
firing rate of a neuron as function of the accumulated evidence variable in a
drift-diffusion model

RETURN
-`tuningcurve`: a vector of floats indicating the firing rate of a neuron as a function of the
latent variable in a drift-diffusion model. Each element corresponds to a discretized and normalized
value of the latent variable, which is typically discretized into 53 bins. The firing rate is in the
unit of spikes/second and is a weighted average across all time step on each trial and across all
trials. The weight corresponds to the moment-to-moment probability of the accumulated evidence. 

ARGUMENT
-`pa`: a nested array whose element `pa{i}{t}{j}` is the probability of the accumulated evidence
variable, `a`, equal to the j-th discretized and normalized value on the t-th time step and on the
i-th trial. The values of the variable `a` is normalized such that the first value corresponds to
`a=-1`, representing the latent variable at the left absorbing bound, and the last value corresponds to
`a=1`, representing `a` being at the right absorbing bound.

-`starttimes_\s`: A vector of floats indicating the time (in seconds) in a recording session to which
the first time step (left edge of the time bin) on each trial of the variable `pa` corresponds. This
argument is used to align `pa` to the spike times

-`spiketimes_s`: a vector of floats indicating all the times (in seconds) in a recording session
when the spikes occurred for a particular neuron
%}
validateattributes(pa, {'cell'}, {'vector'})
validateattributes(pa{1}, {'cell'}, {'vector'})
validateattributes(starttimes_s, {'double'}, {'vector'})
validateattributes(spiketimes_s, {'double'}, {'vector'})
P = inputParser;
addParameter(P, 'dt', 0.01, @(x) isnumeric(x) && isscalar(x))
parse(P, varargin{:});
P = P.Results;
assert(numel(pa)==numel(starttimes_s))
nbins = numel(pa{1}{1});
[tuningcurve, weight] = deal(zeros(nbins,1));
ntrials = numel(pa);
for i = 1:ntrials
    ntimesteps = numel(pa{i});
    timesteps = (0:ntimesteps)*P.dt + starttimes_s(i);
    fr = histcounts(spiketimes_s, timesteps)/P.dt;
    for t = 1:ntimesteps
        for j = 1:nbins
            tuningcurve(j) = tuningcurve(j) + fr(t)*pa{i}{t}(j);
            weight(j) = weight(j) + pa{i}{t}(j);
        end
    end
end
tuningcurve = tuningcurve./weight;