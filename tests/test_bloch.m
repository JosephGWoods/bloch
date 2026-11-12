classdef test_bloch < matlab.unittest.TestCase
% Unit tests for bloch.m
%
% Run from the repo root:
%   addpath('matlab')
%   results = runtests('tests/test_bloch.m')
%
% Physics notation:
%   GAMMA = 2pi rad/Hz (as used by bloch.c)
%   Real  b1 -> rotation around x: Mz=+1 -> My=-1 for 90 deg flip
%   Imag  b1 -> rotation around y: Mz=+1 -> Mx=+1 for 90 deg flip
%   df/gr    -> rotation around z: from Mx=+1: Mx=cos(2pi*f*T), My=sin(2pi*f*T)
%
% Output shape convention (from bloch_mex.c):
%   mode=0: singleton dims removed, remaining dims ordered [nb1, nf, npos, nvel]
%   mode=1: singleton dims removed, remaining dims ordered [ntout, nb1, nf, npos, nvel]

    properties (Constant)
        NTIME = 100
        DT    = 10e-6   % 10 us per step -> 1 ms total
        TOL   = 1e-12   % absolute tolerance for floating-point comparisons
    end

    methods (Static)

        function b1 = hardPulse(flipRad, phaseRad, ntime, dt)
            % Hard-pulse b1 array for a given flip angle (rad) and phase (rad).
            if nargin < 2, phaseRad = 0; end
            if nargin < 3, ntime = test_bloch.NTIME; end
            if nargin < 4, dt = test_bloch.DT; end
            b1_amp = flipRad / (2 * pi * ntime * dt);
            b1 = b1_amp * exp(1j * phaseRad) * ones(ntime, 1);
        end

        function b1 = noRF(ntime)
            if nargin < 1, ntime = test_bloch.NTIME; end
            b1 = zeros(ntime, 1);
        end

        function gr = noGr(ntime)
            if nargin < 1, ntime = test_bloch.NTIME; end
            gr = zeros(ntime, 1);
        end

    end

    methods (Test)

        % ----------------------------------------------------------------
        % Hard-pulse flip-angle tests (no relaxation)
        % ----------------------------------------------------------------

        function test_90deg_x_pulse_my_minus_one(tc)
            % Real (0-phase) 90-flip RF tips Mz->0, My->-1, Mx->0.
            b1 = tc.hardPulse(pi/2);
            [mx, my, mz] = bloch(b1, tc.noGr(), tc.DT);
            tc.verifyEqual(mx, 0.0,  'AbsTol', tc.TOL);
            tc.verifyEqual(my, -1.0, 'AbsTol', tc.TOL);
            tc.verifyEqual(mz, 0.0,  'AbsTol', tc.TOL);
        end

        function test_180deg_x_pulse_mz_minus_one(tc)
            % Real (0-phase) 180-flip RF inverts Mz (Mx->0, My->0).
            b1 = tc.hardPulse(pi);
            [mx, my, mz] = bloch(b1, tc.noGr(), tc.DT);
            tc.verifyEqual(mx, 0.0,  'AbsTol', tc.TOL);
            tc.verifyEqual(my, 0.0,  'AbsTol', tc.TOL);
            tc.verifyEqual(mz, -1.0, 'AbsTol', tc.TOL);
        end

        function test_90deg_y_pulse_mx_plus_one(tc)
            % Imaginary (90-phase) 90-flip RF tips Mz->+Mx.
            b1 = tc.hardPulse(pi/2, pi/2);
            [mx, my, mz] = bloch(b1, tc.noGr(), tc.DT);
            tc.verifyEqual(mx, 1.0, 'AbsTol', tc.TOL);
            tc.verifyEqual(my, 0.0, 'AbsTol', tc.TOL);
            tc.verifyEqual(mz, 0.0, 'AbsTol', tc.TOL);
        end

        function test_zero_flip_no_change(tc)
            % Zero-amplitude (0-flip) RF leaves Mz = 1 unchanged.
            [mx, my, mz] = bloch(tc.noRF(), tc.noGr(), tc.DT);
            tc.verifyEqual(mz, 1.0, 'AbsTol', tc.TOL);
            tc.verifyEqual(mx, 0.0, 'AbsTol', tc.TOL);
            tc.verifyEqual(my, 0.0, 'AbsTol', tc.TOL);
        end

        function test_magnetisation_magnitude_conserved(tc)
            % |M| is conserved (= 1) without relaxation.
            b1 = tc.hardPulse(pi/2);
            [mx, my, mz] = bloch(b1, tc.noGr(), tc.DT);
            tc.verifyEqual(sqrt(mx^2 + my^2 + mz^2), 1.0, 'AbsTol', tc.TOL);
        end

        function test_b1_scaling_reduces_flip_angle(tc)
            % Half-scale RF turns a 90-degree pulse into a 45-degree tip.
            db1 = [1.0; 0.5];
            b1 = tc.hardPulse(pi/2);
            [mx, my, mz] = bloch(b1, tc.noGr(), tc.DT, inf, inf, db1);

            tc.verifyEqual(size(mx), [2, 1]);
            tc.verifyEqual(mx, [0; 0], 'AbsTol', tc.TOL);
            tc.verifyEqual(my, [-1; -sin(pi/4)], 'AbsTol', tc.TOL);
            tc.verifyEqual(mz, [0; cos(pi/4)], 'AbsTol', tc.TOL);
        end

        % ----------------------------------------------------------------
        % Relaxation tests
        % ----------------------------------------------------------------

        function test_t2_decay_no_rf(tc)
            % With no RF, transverse Mx decays as exp(-T/T2).
            t2    = 0.1;        % 100 ms
            ntime = 1000;
            dt    = t2 / ntime; % T = T2
            [mx, ~, ~] = bloch(tc.noRF(ntime), tc.noGr(ntime), dt, inf, t2, 1, 0, 0, 0, 0, ...
                1, 0, 0);
            tc.verifyEqual(mx, exp(-1.0), 'AbsTol', tc.TOL);
        end

        function test_t1_recovery_from_inversion(tc)
            % Mz recovers as 1 - 2*exp(-T/T1) after inversion.
            t1    = 0.1;
            ntime = 1000;
            dt    = t1 / ntime; % T = T1
            [mx, my, mz] = bloch(tc.noRF(ntime), tc.noGr(ntime), dt, t1, inf, 1, 0, 0, 0, 0, ...
                0, 0, -1);
            tc.verifyEqual(mx, 0.0,               'AbsTol', tc.TOL);
            tc.verifyEqual(my, 0.0,               'AbsTol', tc.TOL);
            tc.verifyEqual(mz, 1.0 - 2*exp(-1.0), 'AbsTol', tc.TOL);
        end

        % ----------------------------------------------------------------
        % Off-resonance / gradient precession tests
        % ----------------------------------------------------------------

        function test_off_resonance_precession(tc)
            % Off-resonance: Mx=1 precesses as cos/sin(2pi*df*T).
            df    = 100.0;   % Hz
            ntime = 500;
            dt    = 10e-6;   % 10 us -> 5 ms total
            T     = ntime * dt;
            [mx, my, ~] = bloch(tc.noRF(ntime), tc.noGr(ntime), dt, inf, inf, 1, df, 0, 0, 0, ...
                1, 0, 0);
            theta = 2 * pi * df * T;
            tc.verifyEqual(mx, cos(theta), 'AbsTol', tc.TOL);
            tc.verifyEqual(my, sin(theta), 'AbsTol', tc.TOL);
        end

        function test_gradient_dephasing_single_position(tc)
            % Gradient + position: Mx=1 precesses as cos/sin(2pi*G*x*T).
            gr_amp = 100.0;  % Hz/m
            pos    = 0.5;    % m
            ntime  = 500;
            dt     = 10e-6;  % 10 us -> 5 ms total
            T      = ntime * dt;
            gr     = gr_amp * ones(ntime, 1);
            [mx, my, ~] = bloch(tc.noRF(ntime), gr, dt, inf, inf, 1, 0, pos, 0, 0, ...
                1, 0, 0);
            theta = 2 * pi * gr_amp * pos * T;
            tc.verifyEqual(mx, cos(theta), 'AbsTol', tc.TOL);
            tc.verifyEqual(my, sin(theta), 'AbsTol', tc.TOL);
        end

        function test_gradient_dephasing_multiple_positions(tc)
            % Phase at each position is proportional to position.
            gr_amp = 100.0;
            pos    = [0.0; 0.005; 0.01; 0.02];  % (4x1) column vector
            ntime  = 500;
            dt     = 10e-6;
            T      = ntime * dt;
            gr     = gr_amp * ones(ntime, 1);
            mx0    = ones(length(pos), 1);
            my0    = zeros(length(pos), 1);
            mz0    = zeros(length(pos), 1);
            [mx, my, ~] = bloch(tc.noRF(ntime), gr, dt, inf, inf, 1, 0, pos, 0, 0, ...
                mx0, my0, mz0);
            theta = 2 * pi * gr_amp * pos * T;
            tc.verifyEqual(mx, cos(theta), 'AbsTol', tc.TOL);
            tc.verifyEqual(my, sin(theta), 'AbsTol', tc.TOL);
        end

        function test_off_resonance_multiple_df(tc)
            % Phase at each df is proportional to df.
            df    = [-200.0; -100.0; 0.0; 100.0; 200.0];  % (5x1) column vector
            ntime = 500;
            dt    = 10e-6;
            T     = ntime * dt;
            % mx0 must have npos*nf*nvel = 1*5*1 = 5 elements
            mx0 = ones(length(df), 1);
            my0 = zeros(length(df), 1);
            mz0 = zeros(length(df), 1);
            [mx, my, ~] = bloch(tc.noRF(ntime), tc.noGr(ntime), dt, inf, inf, 1, df, 0, 0, 0, ...
                mx0, my0, mz0);
            theta = 2 * pi * df * T;
            tc.verifyEqual(mx(:), cos(theta(:)), 'AbsTol', tc.TOL);
            tc.verifyEqual(my(:), sin(theta(:)), 'AbsTol', tc.TOL);
        end

        % ----------------------------------------------------------------
        % Initial magnetisation and spoiling
        % ----------------------------------------------------------------

        function test_custom_initial_magnetisation_preserved(tc)
            % Without RF or relaxation, initial M is unchanged.
            mx0_val = 0.5;
            my0_val = 0.3;
            mz0_val = sqrt(1.0 - mx0_val^2 - my0_val^2);
            [mx, my, mz] = bloch(tc.noRF(), tc.noGr(), tc.DT, inf, inf, 1, 0, 0, 0, 0, ...
                mx0_val, my0_val, mz0_val);
            tc.verifyEqual(mx, mx0_val, 'AbsTol', tc.TOL);
            tc.verifyEqual(my, my0_val, 'AbsTol', tc.TOL);
            tc.verifyEqual(mz, mz0_val, 'AbsTol', tc.TOL);
        end

        function test_spoil_zeros_transverse_at_first_step(tc)
            % Spoiling flag at step 1 zeros Mx/My before that step.
            % After spoil with Mz=0: M is all-zero and stays so with no RF.
            spoil    = zeros(tc.NTIME, 1);
            spoil(1) = 1;
            [mx, my, mz] = bloch(tc.noRF(), tc.noGr(), tc.DT, inf, inf, 1, 0, 0, 0, 0, ...
                1, 0, 0, spoil);
            tc.verifyEqual(mx, 0.0, 'AbsTol', tc.TOL);
            tc.verifyEqual(my, 0.0, 'AbsTol', tc.TOL);
            tc.verifyEqual(mz, 0.0, 'AbsTol', tc.TOL);
        end

        function test_spoil_does_not_affect_mz(tc)
            % Spoiling zeroes only transverse components, not Mz.
            spoil    = zeros(tc.NTIME, 1);
            spoil(1) = 1;
            [~, ~, mz] = bloch(tc.noRF(), tc.noGr(), tc.DT, inf, inf, 1, 0, 0, 0, 0, ...
                0, 0, 1, spoil);
            tc.verifyEqual(mz, 1.0, 'AbsTol', tc.TOL);
        end

        % ----------------------------------------------------------------
        % Output shape tests
        %
        % Shape rules (bloch.m):
        %   noutdim==5 -> [ntout, nb1, nf, npos, nvel]
        %   noutdim==4 -> the 4 non-singleton dims in [ntout,nb1,nf,npos,nvel] order
        %   noutdim<=2 -> 2D: first non-singleton dim x 1
        % ----------------------------------------------------------------

        function test_all_scalars_mode0_is_scalar(tc)
            % All scalar inputs, mode=0 -> [1,1] output.
            [mx, my, mz] = bloch(tc.noRF(), tc.noGr(), tc.DT, inf, inf, 1, 0, 0, 0, 0);
            tc.verifyEqual(size(mx), [1, 1]);
            tc.verifyEqual(size(my), [1, 1]);
            tc.verifyEqual(size(mz), [1, 1]);
        end

        function test_all_scalars_mode1_shape_ntime(tc)
            % All scalar inputs, mode=1 -> [NTIME, 1] output.
            [mx, my, mz] = bloch(tc.noRF(), tc.noGr(), tc.DT, inf, inf, 1, 0, 0, 0, 1);
            tc.verifyEqual(size(mx), [tc.NTIME, 1]);
            tc.verifyEqual(size(my), [tc.NTIME, 1]);
            tc.verifyEqual(size(mz), [tc.NTIME, 1]);
        end

        function test_single_element_initial_magnetisation_mode0(tc)
            [mxScalar, myScalar, mzScalar] = bloch(tc.noRF(), tc.noGr(), tc.DT, ...
                inf, inf, 1, 0, 0, 0, 0, 0, 0, 0);
            [mxArray, myArray, mzArray] = bloch(tc.noRF(), tc.noGr(), tc.DT, ...
                inf, inf, 1, 0, 0, 0, 0, [0], [0], [0]);

            tc.verifyEqual(size(mxScalar), [1, 1]);
            tc.verifyEqual(size(myScalar), [1, 1]);
            tc.verifyEqual(size(mzScalar), [1, 1]);
            tc.verifyEqual(mxScalar, mxArray, 'AbsTol', tc.TOL);
            tc.verifyEqual(myScalar, myArray, 'AbsTol', tc.TOL);
            tc.verifyEqual(mzScalar, mzArray, 'AbsTol', tc.TOL);
        end

        function test_single_element_initial_magnetisation_mode1(tc)
            [mxScalar, myScalar, mzScalar] = bloch(tc.noRF(), tc.noGr(), tc.DT, ...
                inf, inf, 1, 0, 0, 0, 1, 0, 0, 0);
            [mxArray, myArray, mzArray] = bloch(tc.noRF(), tc.noGr(), tc.DT, ...
                inf, inf, 1, 0, 0, 0, 1, [0], [0], [0]);

            tc.verifyEqual(size(mxScalar), [tc.NTIME, 1]);
            tc.verifyEqual(size(myScalar), [tc.NTIME, 1]);
            tc.verifyEqual(size(mzScalar), [tc.NTIME, 1]);
            tc.verifyEqual(mxScalar, mxArray, 'AbsTol', tc.TOL);
            tc.verifyEqual(myScalar, myArray, 'AbsTol', tc.TOL);
            tc.verifyEqual(mzScalar, mzArray, 'AbsTol', tc.TOL);
        end

        function test_multi_db1_mode0(tc)
            % nb1=2, mode=0 -> [2, 1]
            db1 = [0.5, 1.0];
            [mx, my, mz] = bloch(tc.noRF(), tc.noGr(), tc.DT, inf, inf, db1);
            tc.verifyEqual(size(mx), [2, 1]);
            tc.verifyEqual(size(my), [2, 1]);
            tc.verifyEqual(size(mz), [2, 1]);
        end

        function test_multi_df_mode0(tc)
            % nf=5, mode=0 -> [5, 1]
            df = linspace(-200, 200, 5)';
            [mx, my, mz] = bloch(tc.noRF(), tc.noGr(), tc.DT, inf, inf, 1, df);
            tc.verifyEqual(size(mx), [5, 1]);
            tc.verifyEqual(size(my), [5, 1]);
            tc.verifyEqual(size(mz), [5, 1]);
        end

        function test_multi_dp_mode0(tc)
            % npos=3, mode=0 -> [3, 1]
            dp = [0.0; 1.0; 2.0];
            [mx, my, mz] = bloch(tc.noRF(), tc.noGr(), tc.DT, inf, inf, 1, 0, dp);
            tc.verifyEqual(size(mx), [3, 1]);
            tc.verifyEqual(size(my), [3, 1]);
            tc.verifyEqual(size(mz), [3, 1]);
        end

        function test_multi_dv_mode0(tc)
            % nvel=3, mode=0 -> [3, 1]
            dv = [0.0; 1.0; 2.0];
            [mx, my, mz] = bloch(tc.noRF(), tc.noGr(), tc.DT, inf, inf, 1, 0, 0, dv);
            tc.verifyEqual(size(mx), [3, 1]);
            tc.verifyEqual(size(my), [3, 1]);
            tc.verifyEqual(size(mz), [3, 1]);
        end

        function test_multi_df_and_dp_mode0(tc)
            % npos=3, nf=5, mode=0 -> [5, 3]
            df = linspace(-200, 200, 5)';
            dp = [0.0; 1.0; 2.0];
            [mx, my, mz] = bloch(tc.noRF(), tc.noGr(), tc.DT, inf, inf, 1, df, dp);
            tc.verifyEqual(size(mx), [5, 3]);
            tc.verifyEqual(size(my), [5, 3]);
            tc.verifyEqual(size(mz), [5, 3]);
        end

        function test_multi_df_dp_dv_mode0(tc)
            % nf=5, npos=3, nvel=2, mode=0 -> [5, 3, 2]
            df = linspace(-200, 200, 5)';
            dp = [0.0; 1.0; 2.0];
            dv = [0.0; 10.0];
            [mx, my, mz] = bloch(tc.noRF(), tc.noGr(), tc.DT, inf, inf, 1, df, dp, dv);
            tc.verifyEqual(size(mx), [5, 3, 2]);
            tc.verifyEqual(size(my), [5, 3, 2]);
            tc.verifyEqual(size(mz), [5, 3, 2]);
        end

        function test_multi_df_dp_dv_mode1(tc)
            % nt=NTIME, nf=5, npos=3, nvel=2, mode=1 -> [NTIME, 5, 3, 2]
            df = linspace(-200, 200, 5)';
            dp = [0.0; 1.0; 2.0];
            dv = [0.0; 10.0];
            [mx, my, mz] = bloch(tc.noRF(), tc.noGr(), tc.DT, inf, inf, 1, df, dp, dv, 1);
            tc.verifyEqual(size(mx), [tc.NTIME, 5, 3, 2]);
            tc.verifyEqual(size(my), [tc.NTIME, 5, 3, 2]);
            tc.verifyEqual(size(mz), [tc.NTIME, 5, 3, 2]);
        end

        function test_multi_db1_df_dp_dv_mode0(tc)
            % db1=2, nf=3, npos=4, nvel=5 -> [2, 3, 4, 2]
            db1 = [0.5; 1.0];
            df = [-100.0; 0.0; 100.0];
            dp = [0.0; 0.01; 0.02; 0.03];
            dv = [0.0; 0.1; 0.2; 0.3; 0.4];
            [mx, my, mz] = bloch(tc.noRF(), tc.noGr(), tc.DT, inf, inf, ...
                db1, df, dp, dv, 0);

            expectedSize = [2, 3, 4, 5];
            tc.verifyEqual(size(mx), expectedSize);
            tc.verifyEqual(size(my), expectedSize);
            tc.verifyEqual(size(mz), expectedSize);
        end

        function test_multi_db1_df_dp_dv_mode1(tc)
            % nt=NTIME, db1=2, nf=3, npos=4, nvel=5.
            db1 = [0.5; 1.0];
            df = [-100.0; 0.0; 100.0];
            dp = [0.0; 0.01; 0.02; 0.03];
            dv = [0.0; 0.1; 0.2; 0.3; 0.4];
            [mx, my, mz] = bloch(tc.noRF(), tc.noGr(), tc.DT, inf, inf, ...
                db1, df, dp, dv, 1);

            expectedSize = [tc.NTIME, 2, 3, 4, 5];
            tc.verifyEqual(size(mx), expectedSize);
            tc.verifyEqual(size(my), expectedSize);
            tc.verifyEqual(size(mz), expectedSize);
        end

        function test_1d_gr(tc)
            gr = 0.5 * ones(tc.NTIME, 1);
            [mx, my, mz] = bloch(tc.noRF(), gr, tc.DT);
            tc.verifyEqual(size(mx), [1, 1]);
            tc.verifyEqual(size(my), [1, 1]);
            tc.verifyEqual(size(mz), [1, 1]);
        end

        function test_2d_gr_xy(tc)
            gr = zeros(tc.NTIME, 2);
            [mx, my, mz] = bloch(tc.noRF(), gr, tc.DT);
            tc.verifyEqual(size(mx), [1, 1]);
            tc.verifyEqual(size(my), [1, 1]);
            tc.verifyEqual(size(mz), [1, 1]);
        end

        function test_3d_gr_xyz(tc)
            gr = zeros(tc.NTIME, 3);
            [mx, my, mz] = bloch(tc.noRF(), gr, tc.DT);
            tc.verifyEqual(size(mx), [1, 1]);
            tc.verifyEqual(size(my), [1, 1]);
            tc.verifyEqual(size(mz), [1, 1]);
        end

        function test_dp_2d_npos_x3(tc)
            dp = zeros(4, 3);
            [mx, my, mz] = bloch(tc.noRF(), tc.noGr(), tc.DT, ...
                inf, inf, 1, 0, dp);
            tc.verifyEqual(size(mx), [4, 1]);
            tc.verifyEqual(size(my), [4, 1]);
            tc.verifyEqual(size(mz), [4, 1]);
        end

        function test_dv_2d_nvel_x3(tc)
            dv = zeros(3, 3);
            [mx, my, mz] = bloch(tc.noRF(), tc.noGr(), tc.DT, ...
                inf, inf, 1, 0, 0, dv);
            tc.verifyEqual(size(mx), [3, 1]);
            tc.verifyEqual(size(my), [3, 1]);
            tc.verifyEqual(size(mz), [3, 1]);
        end

        function test_scalar_tp_broadcasts(tc)
            b1 = tc.hardPulse(pi/2);
            tpArray = tc.DT * ones(tc.NTIME, 1);
            [mxScalar, myScalar, mzScalar] = bloch(b1, tc.noGr(), tc.DT);
            [mxArray, myArray, mzArray] = bloch(b1, tc.noGr(), tpArray);
            tc.verifyEqual(mxArray, mxScalar, 'AbsTol', tc.TOL);
            tc.verifyEqual(myArray, myScalar, 'AbsTol', tc.TOL);
            tc.verifyEqual(mzArray, mzScalar, 'AbsTol', tc.TOL);
        end

    end

end
