function neuronindices  = selectneurons(Cells, Trials, schemename)
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
%   schemename
%       A char or string specifying the name of the scheme for selecting
%       neural neurons
%
%=OUTPUT
%
%   neuronindices
%       A logical index indicating the selected neurons
    validateattributes(schemename, {'char', 'string'},{})
    spreadsheetpath = [fileparts(mfilename('fullpath')), filesep, 'neuron_selection_schemes.csv'];
    T = readtable(spreadsheetpath);
    for variable = string(T.Properties.VariableNames)
        if iscellstr(T.(variable)) && ~strcmp(variable, 'brain_areas')
            T.(variable) = string(T.(variable));
        end
    end
    T.mask_movement = logical(T.mask_movement);
    T.brain_areas = cellfun(@(x) string(split(x, ', '))', T.brain_areas, 'uni', 0);
    indices = find(T.name == schemename);
    ndisjunctions = numel(indices); % inclusive, rather than exclusive, disjunctions
    assert(ndisjunctions>0, 'Cannot find a unique neuron selection scheme named %s', schemename)
    neuronindices = false(length(Cells.raw_spike_time_s),1);
    peakselectivity = zeros(length(Cells.raw_spike_time_s),1);
    for i = indices(:)'
        bin_edges_s = T.t0_s(i) + [0, T.dur_s(i)];
        switch T.criterion{i}
            case "neuronindex"
                isselective = false(size(neuronindices));
                isselective(T.neuronindex(i)) = true;
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
               auROC = choice_auROC(Cells, Trials, ...
                                    'reference_event', T.reference_event(i), ...
                                    'mask_movement', T.mask_movement(i), ...
                                    'bin_edges_s', bin_edges_s);
                selectivity = abs(auROC-0.5);    
                peakselectivity = max(peakselectivity,selectivity);
                FR = firingrate(Cells, Trials, ...
                              'reference_event', T.reference_event(i), ...
                              'time_s_from', T.t0_s(i), ...
                              'time_s_to', T.dur_s(i), ...
                              'trialindices', Trials.responded & Trials.trial_type == 'a');
                FR = cellfun(@mean, FR);
                isselective = selectivity>T.min_abs_choice_auROC(i) & FR>T.min_firing_rate(i);
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
    peakselectivity = peakselectivity(neuronindices);