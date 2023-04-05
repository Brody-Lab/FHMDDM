function log_posterior_curvature(S, varargin)
% plot the dimensions in parameter space with the least or most local
% curvature
%
% ARGUMENT
%
%   `S` is a structure containing the following fields:
%       
%       'hessian_log_posterior'
%       
%       'parameternames'
%
% OPTIONAL ARGUMENT
%
%   prefix
%       a char array that precedes the names of the results files 
%
%   eigenvalues
%       plot the smallest or largest
%
%   minimum_magntude
%       minimum magnitude for the value of an eigenvector along a dimension for that projection to
%       be plotted
%
%   ncolumns
%       number of columns in the plot
%
%   nrows
%       number of rows in the plot
%
%   scalefactor_transformation
%       new scale factor for the transformationinearity parameter in the accumulatormulator
%       transformation
%
%   scalefactor_postspike
%       new scale factor for the temporal basis functions for the
%       post-spike filter
%
%   scalefactor_poststereoclick
%       new scale factor for the temporal basis functions for the
%       post-poststereoclick filter
%
%   scalefactor_premovement
%       new scale factor for the temporal basis functions for the
%       pre-premovementment filter
%
%   scalefactor_postphotostimulus
%       new scale factor for the temporal basis functions for the
%       postphotostimulusostimulus filter
%
%   scalefactor_accumulator
%       new scale factor for the temporal basis functions for the
%       encoding of the transformed mentally accumulatormulated evidence
%
%   shrinkage_latent
%       new L2 shrinkage penalty coefficients for the latent variable
%       parameters
%
%   shrinkage_transformation
%       new L2 shrinkage penalty coefficients for the transformationinearity
%       parameter in the transformation of mentally accumulatormulated evidence
%
%   shrinkage_postspike
%       new L2 shrinkage penalty coefficients for weights of the post-spike
%       temporal basis vectors
%
%   shrinkage_poststereoclick
%       new L2 shrinkage penalty coefficients for weights of the
%       post-stereoclick temporal basis vectors
%
%   shrinkage_premovement
%       new L2 shrinkage penalty coefficients for weights of the
%       pre-premovementment temporal basis vectors
%
%   shrinkage_postphotostimulus
%       new L2 shrinkage penalty coefficients for weights of the
%       postphotostimulusostimulus temporal basis vectors
%
%   shrinkage_accumulator
%       new L2 shrinkage penalty coefficients for weights of the
%       pre-commitment temporal basis vectors
P = inputParser;
addParameter(P, 'prefix', 'results', @(x) ischar(x))
addParameter(P, 'eigenvalues', 'smallest', @(x) ischar(x))
addParameter(P, 'minimum_magnitude', 0.1, @(x) isnumeric(x) && isscalar(x) && x>0)
addParameter(P, 'nrows', 4, @(x) isnumeric(x) && isscalar(x) && x>0)
addParameter(P, 'ncolumns', 2, @(x) isnumeric(x) && isscalar(x) && x>0)
addParameter(P, 'scalefactor_transformation', [], @(x) isnumeric(x) && isscalar(x) && x>0)
addParameter(P, 'scalefactor_gain', [], @(x) isnumeric(x) && isscalar(x) && x>0)
addParameter(P, 'scalefactor_postspike', [], @(x) isnumeric(x) && isscalar(x) && x>0)
addParameter(P, 'scalefactor_poststereoclick', [], @(x) isnumeric(x) && isscalar(x) && x>0)
addParameter(P, 'scalefactor_premovement', [], @(x) isnumeric(x) && isscalar(x) && x>0)
addParameter(P, 'scalefactor_postphotostimulus', [], @(x) isnumeric(x) && isscalar(x) && x>0)
addParameter(P, 'scalefactor_accumulator', [], @(x) isnumeric(x) && isscalar(x) && x>0)
addParameter(P, 'shrinkage_latent', [], @(x) isnumeric(x) && isscalar(x) && x>=0)
addParameter(P, 'shrinkage_transformation', [], @(x) isnumeric(x) && isscalar(x) && x>=0)
addParameter(P, 'shrinkage_gain', [], @(x) isnumeric(x) && isscalar(x) && x>=0)
addParameter(P, 'shrinkage_postspike', [], @(x) isnumeric(x) && isscalar(x) && x>=0)
addParameter(P, 'shrinkage_poststereoclick', [], @(x) isnumeric(x) && isscalar(x) && x>=0)
addParameter(P, 'shrinkage_premovement', [], @(x) isnumeric(x) && isscalar(x) && x>=0)
addParameter(P, 'shrinkage_postphotostimulus', [], @(x) isnumeric(x) && isscalar(x) && x>=0)
addParameter(P, 'shrinkage_accumulator', [], @(x) isnumeric(x) && isscalar(x) && x>=0)
parse(P, varargin{:});
P = P.Results;
%% tabulate parameters
nparameters = numel(S.parameternames);
neuronindices = nan(nparameters,1);
expression1 = 'neuron(\d+)';
parametertypes = strings(nparameters,1);
for i = 1:nparameters
    [~, tokens] = regexp(S.parameternames{i}, expression1, 'match', 'tokens');
    if ~isempty(tokens)
        neuronindices(i) = str2double(tokens{1}{1});
        if contains(S.parameternames{i}, 'gain')
            parametertypes(i) = "gain";
        elseif contains(S.parameternames{i}, 'postspike')
            parametertypes(i) = "postspike";
        elseif contains(S.parameternames{i}, 'premovement')
            parametertypes(i) = "premovement";
        elseif contains(S.parameternames{i}, 'poststereoclick')
            parametertypes(i) = "poststereoclick";
        elseif contains(S.parameternames{i}, 'premovement')
            parametertypes(i) = "premovement";
        elseif contains(S.parameternames{i}, 'postphotostimulus')
            parametertypes(i) = "postphotostimulus";
        elseif contains(S.parameternames{i}, 'overdispersion')
            parametertypes(i) = "overdispersion";
        elseif contains(S.parameternames{i}, 'transformation')
            parametertypes(i) = "transformation";
        elseif contains(S.parameternames{i}, 'commitment')
            parametertypes(i) = "accumulator";
        end
    else
        parametertypes(i) = "latent";
    end
end
T = struct;
T.neuronindex = neuronindices;
T.parametertype = parametertypes;
T = struct2table(T);
neuronlabels = 1:10:max(neuronindices);
thexticks = nan(numel(neuronlabels),1);
for i = 1:numel(neuronlabels)
    thexticks(i) = find(neuronindices==neuronlabels(i),1);
end
%% plot
[V, D] = eig(S.hessian_logposterior);
d = diag(D);
if P.eigenvalues == "smallest"
    sortorder = 'ascend';
else
    sortorder = 'descend';
end
[~, indices] = sort(abs(d), sortorder);
d = d(indices);
V = V(:,indices);
[~,indexmax] = max(abs(d));
dmax = d(indexmax);
big_figure
colors = get(gca, 'colororder');
uniquetypes = unique(T.parametertype);
for i = 1:P.nrows*P.ncolumns
    subplot(P.nrows,P.ncolumns,i)
    fig_prepare_axes
    handles = nan(numel(uniquetypes),1);
    for j = 1:numel(uniquetypes)
        indices = find(T.parametertype == uniquetypes(j));
        v = V(indices,i);
        subindices = abs(v)>P.minimum_magnitude;
        handles(j) = stem(indices(subindices), v(subindices), 'color', colors(j,:));
    end
    if i == 1
        title([P.eigenvalues ' eigenvalue: |\lambda/\lambda_{max}|=' ...
            num2str(abs(d(i)/dmax), '%0.0e')])
    elseif i == 2
        title(['second ' P.eigenvalues ' eigenvalue: |\lambda/\lambda_{max}|=' ...
            num2str(abs(d(i)/dmax), '%0.0e')])
    elseif i == 3
        title(['third ' P.eigenvalues ' eigenvalue: |\lambda/\lambda_{max}|=' ...
            num2str(abs(d(i)/dmax), '%0.0e')])
    else
        title([num2str(i) '-th ' P.eigenvalues ' eigenvalue: |\lambda/\lambda_{max}|=' ...
            num2str(abs(d(i)/dmax), '%0.0e')])
    end
    if i == P.nrows*P.ncolumns
        legend(handles, uniquetypes, 'location', 'best')
    end
    ylabel('eigenvector value')
    ylim([-1,1])
    xlim([0, nparameters+1])
    xlabel('neuron')
    xticks(thexticks)
    xticklabels(arrayfun(@num2str, neuronlabels, 'uni', 0))
end