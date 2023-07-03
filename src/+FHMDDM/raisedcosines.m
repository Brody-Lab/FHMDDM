function bases = raisedcosines(ncosines, nbins, varargin)
% Raised cosines that are compressioned, with increasing width and spacing
% between the peaks of consecutive cosines.
%
%=INPUT
%
% 	ncosines
%       Number of raised cosines
%
%   nbins
%       Number of elements in each cosine
%
%=OPTIONAL INPUT
%
%   compression
%       Positive scalar indicating the degree of compressioning. Larger values
%       indicates more compressioning 
%	
%   begins_at_0
%       Logical scalar indicating whether at the first element, the first
%       cosine is equal to zero or to its maximum value
%
%   ends_at_0
%       Logical scalar indicating whether at the last element, the last
%       cosine is equal to zero or to its maximum value
%
%   zeroindex
%       the bin corresponding to the zero

parser = inputParser;
addParameter(parser, 'compression', NaN,  ...
             @(x) validateattributes(x, {'numeric'}, {'scalar', 'positive'}))
addParameter(parser, 'begins_at_0', false,  ...
             @(x) validateattributes(x, {'logical'}, {'scalar'}))
addParameter(parser, 'ends_at_0', false,  ...
             @(x) validateattributes(x, {'logical'}, {'scalar'}))
addParameter(parser, 'overlap', true,  ...
             @(x) validateattributes(x, {'logical'}, {'scalar'}))
addParameter(parser, 'zeroindex', 1,  ...
             @(x) validateattributes(x, {'numeric'}, {'scalar', 'integer'}))
parse(parser, varargin{:});
P = parser.Results; 
binindinces = (1:nbins) - P.zeroindex;
if ~isnan(P.compression)
    y = asinh(binindinces*P.compression);
else
    y = binindinces;
end
yrange = y(end)-y(1);
if P.begins_at_0 && P.ends_at_0
    Delta_centers = yrange / (ncosines+3);
    centers = y(1)+2*Delta_centers:Delta_centers:y(end)-2*Delta_centers;
elseif P.begins_at_0 && ~P.ends_at_0
    Delta_centers = yrange / (ncosines+1);
    centers = y(1)+2*Delta_centers:Delta_centers:y(end);
elseif ~P.begins_at_0 && P.ends_at_0
    Delta_centers = yrange / (ncosines+1);
    centers = y(1):Delta_centers:y(end)-2*Delta_centers;
else
    Delta_centers = yrange / (ncosines-1);
    centers = y(1):Delta_centers:y(end);
end



if P.overlap
    bases = (cos(max(-pi, min(pi, (y'-centers)*pi/Delta_centers/2))) + 1)/2;
else
    bases = (cos(max(-pi, min(pi, (y'-centers)*pi/Delta_centers))) + 1)/2;
end

bases = bases./max(bases, [], 1);