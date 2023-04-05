function trialset = maketrialset(Cells, Trials, C)
% Process data for one trialset
%
%=ARGUMENT
%
%   Cells
%       A structure containing the spike trains
%
%   Trials
%       A structure containing the behavioral data
%
%   C
%       A structure or a row of table containing parameters for making the trialset
%
%=OUTPUT
%
%   trialset
%       A structure containing the behavioral, stimulus, and neural data of a set of trials
trialindices = FHMDDM.selecttrials(Trials, C.trialselection);
trialset = struct;
trialset.index = 1;
trialset.trials = processtrials(C.maxduration_s, C.timestep_s, Trials, trialindices);
neuronindices = FHMDDM.selectneurons(Cells, Trials, C.neuronselection);
trialset.neurons = FHMDDM.process_spike_trains(Cells, neuronindices, Trials, trialindices);