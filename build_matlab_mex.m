% build_matlab_mex.m — Compile bloch MEX files.
%
% Run this script from the bloch repository root in MATLAB:
%   >> cd /path/to/bloch
%   >> build_matlab_mex
%
% A C compiler must be configured. If needed, run 'mex -setup C' first.

thisDir   = fileparts(mfilename('fullpath'));
matlabDir = fullfile(thisDir, 'matlab');
cDir      = fullfile(thisDir, 'c');

fprintf('Building bloch_mex...\n');
mex(fullfile(matlabDir, 'bloch_mex.c'), ...
    fullfile(cDir,      'bloch.c'),     ...
    '-outdir', matlabDir,               ...
    '-compatibleArrayDims');
fprintf('Done. bloch_mex built in: %s\n', matlabDir);
