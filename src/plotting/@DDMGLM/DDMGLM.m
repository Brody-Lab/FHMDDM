classdef DDMGLM
    methods (Static)
        %%%
        plot_peth(pethset, conditions, varargin)
        %%
        function options = tabulateoptions(analysispath)
            % tabulate the settings of all models fitted in analysis
            %
            % ARGUMENT
            % -'analysispath': absolute path of the folder containing the code and output of the
            % analysis
            %
            % RETURN
            % -`option`: a table
            [~, analysisname] = fileparts(analysispath);
            fitpath = FHMDDM.find_fit_paths(analysisname); 
            options = readtable(fullfile(analysispath, 'options.csv'));
            for field = ["fitname", "recording_id", "trialselection", "unitselection", "objective", ...
                         "choiceobjective", "regression_model_specification"]
                     options.(field) = categorical(options.(field));
            end
            [~, fitnames] = cellfun(@(x) fileparts(char(x)), fitpath, 'uni', 0);
            fitnames = categorical(fitnames);
            [~, sortindex] = intersect(options.fitname, fitnames);
            options = options(sortindex, :);
            options.fitpath = fitpath;
        end
    end
end