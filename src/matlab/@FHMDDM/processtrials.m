function trials = processtrials(maxduration_s, timestep_s, Trials, trialindices)
% Create a trialsetdata object
%
%=INPUT
%
%   maxduration)s
%       maximum duration of the trial, relative to the stereoclick
%
%   timestep_s
%       duration of each time step, in seconds
%
%   Trials
%       A structure containing the behavioral data to be processed
%
%   trialindices
%       A vector of logical indicating which trials are to be included
%
%=OUTPUT
%
%   trials
%       A cell vector whose each element is a struct containing the stimulus and behavioral data of
%       one trial
validateattributes(maxduration_s, {'numeric'},{'scalar', 'positive'})
validateattributes(timestep_s, {'numeric'},{'scalar', 'positive'})
validateattributes(Trials, {'struct'},{})
validateattributes(trialindices, {'logical'},{'vector'})
assert(timestep_s<maxduration_s)
FHMDDM.hasstereoclick(Trials, trialindices)
answer = (Trials.pokedR*2-1).*(Trials.is_hit*2-1);
answer(~Trials.responded) = NaN;
ntrials = sum(trialindices);
previousanswer = zeros(ntrials,1);
trialindices_linear = find(trialindices);
for i = 1:numel(trialindices_linear)
    precedinganswers = answer(1:numel(trialindices_linear(i))-1);
    precedinganswers = precedinganswers(~isnan(precedinganswers));
    if ~isempty(precedinganswers)
        previousanswer(i) = precedinganswers(end);
    end
end
stereoclick_times = get_stereo_click_time(Trials);
trials = cell(ntrials,1);
maxtimesteps = maxduration_s/timestep_s;
for i = 1:ntrials
    index = trialindices_linear(i);
    assert(Trials.leftBups{index}(1)==Trials.rightBups{index}(1))
    trials{i}.clicktimes.L = Trials.leftBups{index} - Trials.leftBups{index}(1);
    trials{i}.clicktimes.R = Trials.rightBups{index} - Trials.rightBups{index}(1);
    trials{i}.gamma = Trials.gamma(index);
    trials{i}.choice = Trials.pokedR(index);
    trials{i}.previousanswer = previousanswer(i);
    trials{i}.movementtime_s = Trials.stateTimes.cpoke_out(index) - stereoclick_times(index);
    trials{i}.ntimesteps = min(maxtimesteps, ceil(trials{i}.movementtime_s/timestep_s));    
    trials{i}.photostimulus_incline_on_s = Trials.stateTimes.laser_on(index) - ...
                                            stereoclick_times(index);
    trials{i}.photostimulus_decline_on_s = Trials.stateTimes.laser_off(index) - ...
                                            stereoclick_times(index);
    trials{i}.stereoclick_time_s = stereoclick_times(index) - ...
        stereoclick_times(linearindices(1));
end