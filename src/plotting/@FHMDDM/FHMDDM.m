classdef FHMDDM
% FHMDDM Factorial hidden Markov drift-diffusion model
    properties
        analysis_data_folder_path_windows = 'X:\tzluo\analysis_data';
    end
    methods (Static)
        R2 = coefficient_of_determination(conditions, indices, peth)
        C = colors
        fit_paths = find_fit_paths(analysisname)
        log_posterior_curvature(S, varargin)
        plot_peth(pethset, condition, time_s, varargin)
        plot_pethset(pethset, time_s, varargin)
        plot_psychometric(choices, Deltaclicks, Echoices, varargin)
        prepareaxes()
        handle = shadeplot(x,lower,upper,varargin)
        startup()
        options = tabulateoptions(analysispath)
    end
end