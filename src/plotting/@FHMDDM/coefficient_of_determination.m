function R2 = coefficient_of_determination(conditions, indices, peth)
%
%
    [SSresidual, SStotal] = deal(0);
    for condition = conditions
        pred = peth.(condition).predicted(indices);
        obsv = peth.(condition).observed(indices);
        SSresidual = SSresidual + sum(((pred - obsv)).^2);
        SStotal = SStotal + sum((mean(obsv) - obsv).^2);
    end
    R2 = 1 - SSresidual/SStotal;
end