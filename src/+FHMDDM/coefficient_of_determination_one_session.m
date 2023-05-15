function R2 = coefficient_of_determination_one_session(outputpath, varargin)
% compute the coefficient of determination for one sesssion
%
% ARGUMENT
%
%   outputpath
%       path of the folder containing the output
%
% RETURN
%
%   R2
%       A vector indicating the coefficient of each neuron
validateattributes('outputpath', {'char'}, {'row'})
parser = inputParser;
addParameter(parser, 'resultsfolder', 'results', @(x) ischar(x))
addParameter(parser, 'conditions', ["leftchoice", "rightchoice"], @(x) isstring(x))

parse(parser, varargin{:});
P = parser.Results; 
PSTH = load(fullfile(outputpath, [P.resultsfolder filesep 'pethsets_stereoclick.mat']));
nneurons = size(PSTH.pethsets{1},1);
time_s = PSTH.time_s;
indices = time_s<=1;
R2 = nan(nneurons,1);
for n = 1:nneurons
    R2(n) = FHMDDM.coefficient_of_determination(P.conditions, indices, PSTH.pethsets{1}{n});
end