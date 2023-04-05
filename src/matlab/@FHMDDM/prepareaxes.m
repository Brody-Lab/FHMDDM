function [] = prepareaxes(varargin)
% PREPAREAXES Personalize an Axes object
%
%   PREPAREAXES() Personalize the current axes. If no axes object exists, one is created.
%
%   PREPAREAXES(A) Personalize a specific Axes object A
if nargin < 1
    a = gca;
else
    a = varargin{:};
end
validateattributes(a, {'matlab.graphics.axis.Axes'}, {'scalar'})
set(a, 'FontName','Arial')
set(a, 'FontSize',14)
set(a, 'ActivePositionProperty', 'outerposition')
set(a, 'Color', 'None')
set(a, 'TickDir', 'out')
set(a, 'NextPlot', 'Add')
set(a, 'LineWidth', 1)
set(a, 'XColor', [0,0,0])
set(a, 'YColor', [0,0,0])
set(a, 'TitleFontWeight', 'normal')
set(a, 'TitleFontSizeMultiplier', 1)
set(a, 'LabelFontSizeMultiplier', 1)