function [] = hasstereoclick(Trials, trialindices)
% Check that the Trials structure contains the time of the stereoclick
first_left_bup = cellfun(@(x) x(1), Trials.leftBups(trialindices));
first_right_bup = cellfun(@(x) x(1), Trials.rightBups(trialindices));
assert(all(abs(first_left_bup-first_right_bup)<eps), 'Missing stereobups')