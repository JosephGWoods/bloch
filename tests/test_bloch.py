"""
pytest unit tests for bloch.bloch().

Run from the repo root:
    pytest tests/test_bloch.py

Physics notation used throughout
---------------------------------
GAMMA = 2π rad/Hz (as used by bloch.c)
Rotation convention (derived from calcrotmat / Cayley-Klein):
  - Real  b1 → rotation around x:  Mz=+1 → My=-1 for 90° flip
  - Imag  b1 → rotation around y:  Mz=+1 → Mx=+1 for 90° flip
  - df/gr    → rotation around z:  from Mx=+1:  Mx=cos(2π·f·T),
                                                My=sin(2π·f·T)
"""
import pytest
import numpy as np

from bloch import bloch

# ---------------------------------------------------------------------------
# Global test parameters and helper functions
# ---------------------------------------------------------------------------
NTIME = 100
DT    = 1e-5   # 10 µs per step → 1 ms total

def _hard_pulse(flip_rad, phase_rad=0.0, ntime=NTIME, dt=DT):
    """Hard-pulse b1 array for a given flip angle (rad) and phase (rad)."""
    b1_amp = flip_rad / (2 * np.pi * ntime * dt)
    return np.full(ntime, b1_amp * np.exp(1j * phase_rad), dtype=complex)

def _no_rf(ntime=NTIME):
    return np.zeros(ntime, dtype=complex)

def _no_gr(ntime=NTIME):
    return np.zeros(ntime)

# ---------------------------------------------------------------------------
# Physics tests
# ---------------------------------------------------------------------------
class TestPhysics:
    """Tests based on simple hard-pulse solutions."""

    # ---- hard-pulse flip-angle tests (no relaxation) -----------------------

    def test_90deg_x_pulse_my_minus_one(self):
        """Real (0°-phase) 90°-flip RF tips Mz → 0, My → -1, Mx → 0."""
        b1 = _hard_pulse(np.pi / 2)
        mx, my, mz = bloch(b1, _no_gr(), DT)
        assert mx == pytest.approx(0.0)
        assert my == pytest.approx(-1.0)
        assert mz == pytest.approx(0.0)

    def test_180deg_x_pulse_mz_minus_one(self):
        """Real (0°-phase) 180°-flip RF inverts Mz (Mx → 0, My → 0)."""
        b1 = _hard_pulse(np.pi)
        mx, my, mz = bloch(b1, _no_gr(), DT)
        assert mx == pytest.approx(0.0)
        assert my == pytest.approx(0.0)
        assert mz == pytest.approx(-1.0)

    def test_90deg_y_pulse_mx_plus_one(self):
        """Imaginary (90°-phase) 90°-flip RF tips Mz → +Mx."""
        b1 = _hard_pulse(np.pi / 2, phase_rad=np.pi / 2)  # pure imaginary
        mx, my, mz = bloch(b1, _no_gr(), DT)
        assert mx == pytest.approx(1.0)
        assert my == pytest.approx(0.0)
        assert mz == pytest.approx(0.0)

    def test_zero_flip_no_change(self):
        """Zero-amplitude (0°-flip) RF leaves Mz = 1 unchanged."""
        mx, my, mz = bloch(_no_rf(), _no_gr(), DT)
        assert mz == pytest.approx(1.0)
        assert mx == pytest.approx(0.0)
        assert my == pytest.approx(0.0)

    def test_magnetisation_magnitude_conserved(self):
        """|M| is conserved (= 1) without relaxation."""
        b1 = _hard_pulse(np.pi / 2)
        mx, my, mz = bloch(b1, _no_gr(), DT)
        assert np.sqrt(mx**2 + my**2 + mz**2) == pytest.approx(1.0)

    # ---- relaxation tests --------------------------------------------------

    def test_t2_decay_no_rf(self):
        """With no RF, transverse Mx decays as exp(-T/T2)."""
        t2 = 0.1       # 100 ms
        ntime = 1000
        dt = t2 / ntime # T = T2
        mx, _, _ = bloch(_no_rf(ntime), _no_gr(ntime), dt, t2=t2, mx0=1.0, my0=0.0, mz0=0.0)
        assert mx == pytest.approx(np.exp(-1.0))

    def test_t1_recovery_from_inversion(self):
        """Mz recovers as 1 - 2·exp(-T/T1) after inversion."""
        t1 = 0.1
        ntime = 1000
        dt = t1 / ntime # T = T1
        mx, my, mz = bloch(_no_rf(ntime), _no_gr(ntime), dt, t1=t1, mx0=0.0, my0=0.0, mz0=-1.0)
        assert mx == pytest.approx(0.0)
        assert my == pytest.approx(0.0)
        assert mz == pytest.approx(1.0 - 2.0 * np.exp(-1.0))

    # ---- off-resonance / gradient precession tests -------------------------

    def test_off_resonance_precession(self):
        """Off-resonance: Mx=1 precesses as cos/sin(2π·df·T)."""
        df = 100.0       # Hz
        ntime = 500
        dt = 10e-6       # 10 us -> 5 ms total
        T = ntime * dt
        mx, my, _ = bloch(_no_rf(ntime), _no_gr(ntime), dt, df=df, mx0=1.0, my0=0.0, mz0=0.0)
        theta = 2 * np.pi * df * T
        assert mx == pytest.approx(np.cos(theta))
        assert my == pytest.approx(np.sin(theta))

    def test_gradient_dephasing_single_position(self):
        """Gradient + position: Mx=1 precesses as cos/+sin(2π·G·x·T)."""
        gr_amp = 100.0   # Hz/m
        pos = 0.5        # m
        ntime = 500
        dt = 10e-6       # 10 us -> 5 ms total
        T = ntime * dt
        gr = np.full(ntime, gr_amp)
        mx, my, _ = bloch(_no_rf(ntime), gr, dt, dp=pos, mx0=1.0, my0=0.0, mz0=0.0)
        theta = 2 * np.pi * gr_amp * pos * T
        assert mx == pytest.approx(np.cos(theta))
        assert my == pytest.approx(np.sin(theta))

    def test_gradient_dephasing_multiple_positions(self):
        """Phase at each position is proportional to position."""
        gr_amp = 100.0     # Hz/m
        pos = np.array([0.0, 0.005, 0.01, 0.02]) # m
        ntime = 500
        dt = 10e-6       # 10 us -> 5 ms total
        total = ntime * dt
        gr = np.full(ntime, gr_amp)
        mx0 = np.ones((1, 1, len(pos)))
        my0 = np.zeros_like(mx0)
        mz0 = np.zeros_like(mx0)
        mx, my, _ = bloch(_no_rf(ntime), gr, dt, dp=pos, mx0=mx0, my0=my0, mz0=mz0)
        theta = 2 * np.pi * gr_amp * pos * total
        assert mx == pytest.approx(np.cos(theta))
        assert my == pytest.approx(np.sin(theta))

    def test_off_resonance_multiple_df(self):
        """Phase at each df is proportional to df."""
        df = np.array([-200.0, -100.0, 0.0, 100.0, 200.0])
        ntime = 500
        dt = 10e-6       # 10 us -> 5 ms total
        total = ntime * dt
        mx0 = np.ones((1, len(df), 1))
        my0 = np.zeros_like(mx0)
        mz0 = np.zeros_like(mx0)
        mx, my, _ = bloch(_no_rf(ntime), _no_gr(ntime), dt, df=df, mx0=mx0, my0=my0, mz0=mz0)
        theta = 2 * np.pi * df * total
        assert mx == pytest.approx(np.cos(theta))
        assert my == pytest.approx(np.sin(theta))

    # ---- initial magnetisation and spoiling --------------------------------

    def test_custom_initial_magnetisation_preserved(self):
        """Without RF or relaxation, initial M is unchanged."""
        mx0_val = 0.5
        my0_val = 0.3
        mz0_val = np.sqrt(1.0 - mx0_val**2 - my0_val**2)
        mx, my, mz = bloch(_no_rf(), _no_gr(), DT, mx0=mx0_val, my0=my0_val, mz0=mz0_val)
        assert mx == pytest.approx(mx0_val)
        assert my == pytest.approx(my0_val)
        assert mz == pytest.approx(mz0_val)

    def test_spoil_zeros_transverse_at_first_step(self):
        """Spoiling flag at step 0 zeros Mx/My before that step."""
        spoil = np.zeros(NTIME)
        spoil[0] = 1
        mx, my, mz = bloch(_no_rf(), _no_gr(), DT, mx0=1.0, my0=0.0, mz0=0.0, spoil=spoil)
        # After spoil: Mx=My=0, Mz=0; no RF → stays all-zero
        assert mx == pytest.approx(0.0)
        assert my == pytest.approx(0.0)
        assert mz == pytest.approx(0.0)

    def test_spoil_does_not_affect_mz(self):
        """Spoiling zeroes only transverse components, not Mz."""
        spoil = np.zeros(NTIME)
        spoil[0] = 1
        mx, my, mz = bloch(_no_rf(), _no_gr(), DT, mx0=0.0, my0=0.0, mz0=1.0, spoil=spoil)
        assert mz == pytest.approx(1.0)

# ---------------------------------------------------------------------------
# Output shape tests
# ---------------------------------------------------------------------------
class TestOutputShape:
    """Tests for correct output shape with various input combinations."""

    # ---- scalar inputs → squeezed to 0-d ----------------------------------

    def test_all_scalars_mode0_is_scalar(self):
        mx, my, mz = bloch(_no_rf(), _no_gr(), DT, mode=0)
        assert mx.ndim == 0
        assert my.ndim == 0
        assert mz.ndim == 0

    def test_all_scalars_mode1_shape_ntime(self):
        mx, my, mz = bloch(_no_rf(), _no_gr(), DT, mode=1)
        assert mx.shape == (NTIME,)
        assert my.shape == (NTIME,)
        assert mz.shape == (NTIME,)

    # ---- single-element arrays also squeeze to 0-d -------------------------

    def test_single_element_df_dp_dv_arrays_mode0_squeezed(self):
        mx_s, my_s, mz_s = bloch(_no_rf(), _no_gr(), DT,
                                 mx0=0.0, my0=0.0, mz0=0.0)
        mx_a, my_a, mz_a = bloch(_no_rf(), _no_gr(), DT,
                                 mx0=np.array([0.0]), my0=np.array([0.0]), mz0=np.array([0.0]))
        assert mx_s.ndim == 0
        assert my_s.ndim == 0
        assert mz_s.ndim == 0
        assert mx_s == mx_a
        assert my_s == my_a
        assert mz_s == mz_a

    def test_single_element_df_dp_dv_arrays_mode1_squeezed(self):
        mx_s, my_s, mz_s = bloch(_no_rf(), _no_gr(), DT,
                                 mx0=0.0, my0=0.0, mz0=0.0, mode=1)
        mx_a, my_a, mz_a = bloch(_no_rf(), _no_gr(), DT,
                                 mx0=np.array([0.0]), my0=np.array([0.0]), mz0=np.array([0.0]), mode=1)
        assert mx_s.shape == (NTIME,)
        assert my_s.shape == (NTIME,)
        assert mz_s.shape == (NTIME,)
        assert mx_s == pytest.approx(mx_a)
        assert my_s == pytest.approx(my_a)
        assert mz_s == pytest.approx(mz_a)

    # ---- multi-element arrays ------------------------------------------

    def test_multi_df_mode0(self):
        df = np.linspace(-200, 200, 5)
        mx, my, mz = bloch(_no_rf(), _no_gr(), DT, df=df)
        assert mx.shape == (5,)
        assert my.shape == (5,)
        assert mz.shape == (5,)

    def test_multi_dp_mode0(self):
        dp = np.array([0.0, 1.0, 2.0])
        mx, my, mz = bloch(_no_rf(), _no_gr(), DT, dp=dp)
        assert mx.shape == (3,)
        assert my.shape == (3,)
        assert mz.shape == (3,)

    def test_multi_dv_mode0(self):
        dv = np.array([0.0, 1.0, 2.0])
        mx, my, mz = bloch(_no_rf(), _no_gr(), DT, dv=dv)
        assert mx.shape == (3,)
        assert my.shape == (3,)
        assert mz.shape == (3,)

    def test_multi_df_and_dp_mode0(self):
        df = np.linspace(-200, 200, 5)
        dp = np.array([0.0, 1.0, 2.0])
        mx, my, mz = bloch(_no_rf(), _no_gr(), DT, df=df, dp=dp)
        assert mx.shape == (5, 3)
        assert my.shape == (5, 3)
        assert mz.shape == (5, 3)

    def test_multi_dv_df_dp_mode0(self):
        dv = np.array([0.0, 10.0])
        df = np.linspace(-200, 200, 5)
        dp = np.array([0.0, 1.0, 2.0])
        mx, my, mz = bloch(_no_rf(), _no_gr(), DT, df=df, dp=dp, dv=dv)
        assert mx.shape == (2, 5, 3)
        assert my.shape == (2, 5, 3)
        assert mz.shape == (2, 5, 3)

    def test_multi_dv_df_dp_mode1(self):
        dv = np.array([0.0, 10.0])
        df = np.linspace(-200, 200, 5)
        dp = np.array([0.0, 1.0, 2.0])
        mx, my, mz = bloch(_no_rf(), _no_gr(), DT, df=df, dp=dp, dv=dv, mode=1)
        assert mx.shape == (2, 5, 3, NTIME)
        assert my.shape == (2, 5, 3, NTIME)
        assert mz.shape == (2, 5, 3, NTIME)

    # ---- gradient array shapes -----------------------------------------

    def test_1d_gr(self):
        gr = np.ones(NTIME) * 0.5
        mx, my, mz = bloch(_no_rf(), gr, DT)
        assert mx.ndim == 0
        assert my.ndim == 0
        assert mz.ndim == 0

    def test_2d_gr_xy(self):
        gr = np.zeros((NTIME, 2))
        mx, my, mz = bloch(_no_rf(), gr, DT)
        assert mx.ndim == 0
        assert my.ndim == 0
        assert mz.ndim == 0

    def test_3d_gr_xyz(self):
        gr = np.zeros((NTIME, 3))
        mx, my, mz = bloch(_no_rf(), gr, DT)
        assert mx.ndim == 0
        assert my.ndim == 0
        assert mz.ndim == 0

    # ---- dp / dv with multiple components ---------------------------------

    def test_dp_2d_npos_x3(self):
        dp = np.zeros((4, 3))   # 4 positions, xyz columns
        mx, my, mz = bloch(_no_rf(), _no_gr(), DT, dp=dp)
        assert mx.shape == (4,)
        assert my.shape == (4,)
        assert mz.shape == (4,)

    def test_dv_2d_nvel_x3(self):
        dv = np.zeros((3, 3))   # 3 velocities, xyz columns
        mx, my, mz = bloch(_no_rf(), _no_gr(), DT, dv=dv)
        assert mx.shape == (3,)
        assert my.shape == (3,)
        assert mz.shape == (3,)

    # ---- tp broadcasting ---------------------------------------------------

    def test_scalar_tp_broadcasts(self):
        """Scalar tp gives the same result as an array of identical values."""
        b1 = _hard_pulse(np.pi / 2)
        tp_arr = np.full(NTIME, DT)
        mx_s, my_s, mz_s = bloch(b1, _no_gr(), DT)
        mx_a, my_a, mz_a = bloch(b1, _no_gr(), tp_arr)
        assert mx_a == pytest.approx(mx_s)
        assert my_a == pytest.approx(my_s)
        assert mz_a == pytest.approx(mz_s)
