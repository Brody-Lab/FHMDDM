function plot_peth(pethset, conditions, varargin)
%PLOT_PETH Plot the peri-event time histogram (PETH) of one neuron for a set of conditions
%   PLOT_PETH(PETHSET, CONDITIONS) plots the PETH in each task condition specified by strings array
%   CONDITIONS in the same axes. The values of the PETH's are containined in the structure PETHSET.
%
%   PLOT_PETH(...,PARAM1,VAL1,PARAM2,VAL2,...) specifies one or more of
%   the following name/value pairs:
%
%       `Ax`        The axes in which the plots are made
%
%       'Colors'    A three-column matrix of RGB triplets. The i-th row corresponds to the i-th
%                   condition task condition specified by CONDITIONS
%
%   