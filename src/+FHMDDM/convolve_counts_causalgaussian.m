function w = convolve_counts_causalgaussian(counts, width, sigma)
% Convolution of binned counts with a causal Gaussian filter
%
% The causal Gaussian filter is normalized by its sum.
%
%=INPUT
%
%   counts
%       A vector whose element is the number of counts in each bin
%
%   width
%       Positive integer indicating the width of the causal
%       Gaussian filter, in units of bins
%
%   sigma
%       Positive integer indicating the standard deviation of the Gaussian
%       filter, in units of bins
%
%=OUTPUT
%       
%   w
%       A vector representing the convolution
validateattributes(counts, {'logical', 'numeric'}, {'vector'})
validateattributes(width, {'numeric'}, {'scalar', 'positive', 'integer'})
validateattributes(sigma, {'numeric'}, {'scalar', 'positive', 'integer'})

causalgaussian = normpdf(0:width-1, 0, sigma);
causalgaussian = causalgaussian/sum(causalgaussian);

w = conv(counts, causalgaussian);
weight = conv(ones(size(counts)), causalgaussian);

w = w./weight;
w = w(1:numel(counts));