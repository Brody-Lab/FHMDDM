function R2 = coefficient_of_determination(conditions, indices, peth)
% COEFFICIENT_OF_DETERMINATION assess the goodness-of-fit of a neuron's peri-event time histogram
%   R2 = COEFFICIENT_OF_DETERMINATION(CONDITIONS, INDICES, PETH) computes the coefficient of
%   determination for the peri-event time histogram in the structure PETH.
%   
%   The time steps contributing to the coefficient of determination are specified by the integer
%   vector INDICES.
%   
%   The goodness-of-fit metric is based on the errors summed across each condition in the string
%   vector CONDITIONS.
validateattributes(conditions, {'string'},{'vector'})
validateattributes(peth, {'struct'},{'scalar'})
validateattributes(indices, {'numeric', 'logical'}, {'vector'})
[SSresidual, SStotal] = deal(0);
for condition = conditions
    n = min(numel(peth.(condition).predicted), numel(peth.(condition).observed));
    if n < numel(indices)
        indices_condition = indices(1:n);
    else
        indices_condition = indices;
    end
    pred = peth.(condition).predicted(indices_condition);
    obsv = peth.(condition).observed(indices_condition);
    SSresidual = SSresidual + sum(((pred - obsv)).^2);
    SStotal = SStotal + sum((mean(obsv) - obsv).^2);
end
R2 = 1 - SSresidual/SStotal;