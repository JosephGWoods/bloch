% build_matlab_mex.m — Compile bloch MEX files.
%
% Run this script from the bloch repository root in MATLAB:
%   >> cd /path/to/bloch
%   >> build_matlab_mex
%
% A C compiler must be configured. If needed, run 'mex -setup C' first.
%
% OpenMP flags are compiler-dependent, so the selected compiler is detected
% and the appropriate flags are chosen for Windows (MSVC or MinGW), Linux
% (gcc/clang) and macOS (clang, requires libomp: 'brew install libomp').

thisDir   = fileparts(mfilename('fullpath'));
matlabDir = fullfile(thisDir, 'matlab');
cDir      = fullfile(thisDir, 'c');

srcFiles = {fullfile(matlabDir, 'bloch_mex.c'), fullfile(cDir, 'bloch.c')};

cc = mex.getCompilerConfigurations('C', 'Selected');
isMSVC = ispc && contains(cc.Manufacturer, 'Microsoft', 'IgnoreCase', true);

fprintf('Building bloch_mex...\n');
if isMSVC
    % MSVC uses /openmp instead of GCC/Clang-style -fopenmp.
    mex('COMPFLAGS="$COMPFLAGS /openmp"', ...
        'OPTIMFLAGS="$OPTIMFLAGS /O2"', ...
        srcFiles{:}, ...
        '-outdir', matlabDir, ...
        '-compatibleArrayDims');
elseif ismac
    % Apple clang needs -Xpreprocessor -fopenmp and libomp linked explicitly.
    mex('CFLAGS="$CFLAGS -Xpreprocessor -fopenmp"', ...
        'LDFLAGS="$LDFLAGS -lomp"', ...
        'COPTIMFLAGS="$COPTIMFLAGS -O3"', ...
        srcFiles{:}, ...
        '-outdir', matlabDir, ...
        '-compatibleArrayDims');
else
    % Linux gcc/clang, and Windows MinGW, both accept -fopenmp directly.
    mex('CFLAGS="$CFLAGS -fopenmp"', ...
        'LDFLAGS="$LDFLAGS -fopenmp"', ...
        'COPTIMFLAGS="$COPTIMFLAGS -O3 -fopenmp"', ...
        srcFiles{:}, ...
        '-outdir', matlabDir, ...
        '-compatibleArrayDims');
end
fprintf('Done. bloch_mex built in: %s\n', matlabDir);
