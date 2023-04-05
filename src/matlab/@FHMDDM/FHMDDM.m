classdef FHMDDM
% FHMDDM Factorial hidden Markov drift-diffusion model
    properties
        analysis_data_folder_path_windows = 'X:\tzluo\analysis_data';
    end
    methods (Static)
        R2 = coefficient_of_determination(conditions, indices, peth)
        C = colors
        fit_paths = find_fit_paths(analysisname)
        hasstereoclick(Trials, trialindices)
        log_posterior_curvature(S, varargin)
        trialset = maketrialset(Cells, Trials, configuration)
        plot_peth(pethset, condition, time_s, varargin)
        plot_pethset(pethset, time_s, varargin)
        plot_psychometric(choices, Deltaclicks, Echoices, varargin)
        S = predict_baseline_basis_functions(spiketimes_s, trialstart_s, trialend_s, varargin)
        S = predict_baseline_by_convolving(spiketimes_s, trialstart_s, trialend_s, varargin)
        prepareaxes()
        processdata(configurationame)
        trials = processtrials(Trials, timestep_s)
        neuronindices  = selectneurons(Cells, Trials, configurationname)
        trialindices = selecttrials(Trials, configurationname)
        handle = shadeplot(x,lower,upper,varargin)
        startup()
        options = tabulateoptions(analysispath)
    end
end