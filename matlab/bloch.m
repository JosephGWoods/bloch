%   Bloch simulator working in Hz units (to treat all nuclei equally).
%
%   [mx,my,mz] = bloch(b1,gr,tp)
%   [mx,my,mz] = bloch(b1,gr,tp,t1,t2,df,dp,dv)
%   [mx,my,mz] = bloch(b1,gr,tp,t1,t2,df,dp,dv,mode)
%   [mx,my,mz] = bloch(b1,gr,tp,t1,t2,df,dp,dv,mode,mx0,my0,mz0)
%   [mx,my,mz] = bloch(b1,gr,tp,t1,t2,df,dp,dv,mode,mx0,my0,mz0,spoil)
%
%   Bloch simulation of rotations due to B1, gradient and off-resonance,
%   including relaxation effects.
%
%   Required inputs:
%       b1   - (Tx1) RF pulse (Hz). Can be complex.
%       gr   - (Tx1,2,3) Gradient waveform (Hz/units). 1, 2, or 3 columns for x[,y[,z]].
%       tp   - (Tx1) Duration of each time step (s).
%              A scalar is broadcast to all T time points.
%              A monotonically increasing vector is interpreted as end times.
%
%   Optional inputs (use [] to accept the default):
%       t1          - T1 relaxation time (s). Default: inf (no T1 relaxation).
%       t2          - T2 relaxation time (s). Default: inf (no T2 relaxation).
%       df          - (Fx1) Off-resonance frequencies (Hz). Default: 0.
%       dp          - (Px1,2,3) Spatial positions (units). Default: 0.
%       dv          - (Vx1,2,3) Velocities (units/s). Default: 0.
%       mode        - 0 = endpoint only (default), 1 = all time points.
%       mx0,my0,mz0 - (PxFxV) Initial magnetisation. Default: [0;0;1].
%       spoil       - (Mx1) Spoiling flags (1 = zero Mx/My before that step).
%                     Default: all zeros.
%
%   Outputs:
%       mx,my,mz - Magnetisation components.
%                  Shape (PxFxV) for mode=0, or (TxPxFxV) for mode=1.
%                  Singleton dimensions are squeezed.
%
%   Sign convention: a 90-degree x-pulse (real B1) rotates Mz to -My.
%   See M. Levitt. "Basics of Nuclear Magnetic Resonance". Page 250.
%
%   B. Hargreaves   Nov 2003.
%   M. Robson       Reversed sign of gyromagnetic ratio.
%   C. Rodgers      Feb 2013. Hz units; debug flag.
%   W. Clarke       Perfect spoiler support.
%   J.G. Woods      Apr 2020. Velocity/flow support.
%                   May 2026. Wrapper only; core code in bloch_mex.c / bloch.c.

function [mx, my, mz] = bloch(b1, gr, tp, t1, t2, df, dp, dv, mode, mx0, my0, mz0, spoil)

% Require the compiled MEX file — does not attempt auto-compilation.
if exist('bloch_mex', 'file') ~= 3
    error('bloch:NotCompiled', ...
        'bloch_mex MEX file not found for this platform. Run build_matlab_mex.m from the bloch repository root.');
end

% Apply defaults for optional parameters
if nargin < 4  || isempty(t1),   t1   = inf; end
if nargin < 5  || isempty(t2),   t2   = inf; end
if nargin < 6  || isempty(df),   df   = 0;   end
if nargin < 7  || isempty(dp),   dp   = 0;   end
if nargin < 8  || isempty(dv),   dv   = 0;   end
if nargin < 9  || isempty(mode), mode = 0;   end
if nargin < 10, mx0   = []; end
if nargin < 11, my0   = []; end
if nargin < 12, mz0   = []; end
if nargin < 13, spoil = []; end

% Build argument list, appending optional trailing args only when needed.
args = {b1, gr, tp, t1, t2, df, dp, dv, mode};

if ~isempty(mx0) || ~isempty(my0) || ~isempty(mz0)
    % Empty matrices fall through to the [0;0;1] default inside the MEX.
    args = [args, {mx0, my0, mz0}];
    if ~isempty(spoil)
        args = [args, {spoil}];
    end
elseif ~isempty(spoil)
    % Spoil requested but no initial magnetisation: pass empty placeholders so
    % the MEX receives nrhs==13 and applies its own [0;0;1] default.
    args = [args, {[], [], [], spoil}];
end

[mx, my, mz] = bloch_mex(args{:});
