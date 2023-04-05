function neuronindices  = selectneurons(Cells, Trials, configurationname)
% Select neurons according to a registered scheme
%
%=INPUT
%
%   Cells
%       A structure made by NP_make_Cells.m
%
%   Trials
%       A structure made by PB_process_data.m
%
%   configurationname
%       A char or string specifying the name of the scheme for selecting
%       neural neurons
%
%=OUTPUT
%
%   neuronindices
%       A logical index indicating the selected neurons
validateattributes(configurationname, {'char', 'string'},{})
spreadsheetpath = [fileparts(mfilename('fullpath')), filesep 'configurations'  filesep, ...
    'neuronselection.csv'];
T = readtable(spreadsheetpath);
for variable = string(T.Properties.VariableNames)
    if iscellstr(T.(variable)) && ~strcmp(variable, 'brain_areas')
        T.(variable) = string(T.(variable));
    end
end
T.mask_movement = logical(T.mask_movement);
T.brain_areas = cellfun(@(x) string(split(x, ', '))', T.brain_areas, 'uni', 0);
indices = find(T.name == configurationname);
ndisjunctions = numel(indices); % inclusive, rather than exclusive, disjunctions
assert(ndisjunctions>0, 'Cannot find a unique unit selection scheme named %s', configurationname)
neuronindices = false(length(Cells.raw_spike_time_s),1);
peakselectivity = zeros(length(Cells.raw_spike_time_s),1);
for i = indices(:)'
    bin_edges_s = T.t0_s(i) + [0, T.dur_s(i)];
    switch T.criterion{i}
        case "unitindex"
            isselective = false(size(neuronindices));
            isselective(T.unitindex(i)) = true;
        case "choice_auROC"
            auROC = choice_auROC(Cells, Trials, ...
                                'reference_event', T.reference_event(i), ...
                                'mask_movement', T.mask_movement(i), ...
                                'bin_edges_s', bin_edges_s);
            selectivity = abs(auROC-0.5);    
            peakselectivity = max(peakselectivity,selectivity);
            isselective = selectivity>T.min_abs_choice_auROC(i);
            if sum(isselective) > T.maxneurons(i)
                [~, I] = sort(selectivity, 'descend');
                isselective(I(T.maxneurons(i)+1:end)) = false;            
            end
        case "firing_rate"
            FR = firingrate(Cells, Trials, ...
                          'reference_event', T.reference_event(i), ...
                          'time_s_from', T.t0_s(i), ...
                          'time_s_to', T.dur_s(i), ...
                          'trialindices', Trials.responded & Trials.trial_type == 'a');
            FR = cellfun(@mean, FR);
            isselective = FR>T.min_firing_rate(i);
            if sum(isselective) > T.maxneurons(i)
                [~, I] = sort(FR, 'descend');
                isselective(I(T.maxneurons(i)+1:end)) = false;            
            end
        case "choice_auROC_fr"
           [auROC, FR] = choice_auROC(Cells, Trials, ...
                                'reference_event', T.reference_event(i), ...
                                'mask_movement', T.mask_movement(i), ...
                                'bin_edges_s', bin_edges_s);
            selectivity = abs(auROC-0.5);    
            peakselectivity = max(peakselectivity,selectivity);
            meanfr = mean([cellfun(@mean, FR.L{1}), cellfun(@mean, FR.R{1})],2);
            isselective = selectivity>T.min_abs_choice_auROC(i) & meanfr>T.min_firing_rate(i);
            if sum(isselective) > T.maxneurons(i)
                [~, I] = sort(selectivity, 'descend');
                isselective(I(T.maxneurons(i)+1:end)) = false;            
            end 
        otherwise
            error('unrecognized selection criterion')
    end
    if T.brain_areas{i}==""
        inbrainarea = true(numel(neuronindices),1);
    else
        inbrainarea = ismember(Cells.cell_area, T.brain_areas{i});
    end
    neuronindices(isselective & inbrainarea) = true;
end
%% exclude nosiy neurons
minDeltaLL = max(T.minDeltaLL(indices));
minmeanDeltaLL = max(T.minmeanDeltaLL(indices));
stereoclick_time_s = get_stereo_click_time(Trials);
trialindices = Trials.responded & ...
                    Trials.trial_type=='a' & ...
                    ~Trials.laser.isOn;
trialstart_s = stereoclick_time_s(trialindices);
choices = Trials.pokedR(trialindices);
nneurons = sum(neuronindices);
neuronindices_linear = find(neuronindices);
DeltaLL = nan(size(neuronindices));
for n = 1:nneurons
    neuron = neuronindices_linear(n);
    spiketimes_s = Cells.raw_spike_time_s{neuron};
    DeltaLL(neuron) = glmscreen(choices, spiketimes_s, trialstart_s);
    if DeltaLL(neuron) < minDeltaLL
        neuronindices(neuron) = false;
    end
end 
meanDeltaLL = mean(DeltaLL(neuronindices));
if meanDeltaLL < minmeanDeltaLL
    neuronindices_linear = find(neuronindices);
    [sortedDeltaLL, sortedindices] = sort(DeltaLL(neuronindices));
    themeans = nan(numel(sortedDeltaLL),1);
    for i = 1:numel(sortedDeltaLL)
        themeans(i) = mean(sortedDeltaLL(i:end));
    end
    i = find(themeans > minmeanDeltaLL, 1);
    neuronindices(neuronindices_linear(sortedindices(1:i-1))) = false;
end