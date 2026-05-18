# pybloch_c.py - Python ctypes wrapper for bloch.c
"""
Python wrapper for the Bloch equation simulator.

Wraps blochsimfz() from bloch.c via ctypes, providing a high-level bloch()
function whose interface mirrors the MATLAB bloch.m wrapper.

Units (must be mutually consistent):
    b1     - RF pulse amplitude (Hz)
    gr     - gradient amplitudes (Hz/units)
    tp     - time steps (s)
    t1, t2 - relaxation times (s)
    df     - off-resonance frequencies (Hz)
    dp     - spatial positions (units)
    dv     - velocities (units/s)

Author: Joseph G. Woods
"""

import numpy as np
import ctypes
import os
import platform

# ---------------------------------------------------------------------------
# Load shared library
# ---------------------------------------------------------------------------
_system = platform.system()
if _system == 'Darwin':
    _libname = 'libbloch.dylib'
elif _system == 'Windows':
    _libname = 'bloch.dll'
elif _system == 'Linux':
    _libname = 'libbloch.so'
else:
    raise OSError(f"Unsupported platform: {_system}")

_lib_path = os.path.join(os.path.dirname(__file__), _libname)
if not os.path.exists(_lib_path):
    raise FileNotFoundError(
        f"Bloch library not found at {_lib_path}. "
        f"Please compile bloch.c first using: bash build_python_lib.sh"
    )

_lib = ctypes.CDLL(_lib_path)

# ---------------------------------------------------------------------------
# Define C function signature for blochsimfz
# ---------------------------------------------------------------------------
_nd = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')

_lib.blochsimfz.argtypes = [
    _nd,             # b1real
    _nd,             # b1imag
    _nd,             # xgrad
    _nd,             # ygrad
    _nd,             # zgrad
    _nd,             # tsteps
    ctypes.c_int,    # ntime
    ctypes.c_double, # t1
    ctypes.c_double, # t2
    _nd,             # dfreq
    ctypes.c_int,    # nfreq
    _nd,             # dxpos
    _nd,             # dypos
    _nd,             # dzpos
    ctypes.c_int,    # npos
    _nd,             # dxvel
    _nd,             # dyvel
    _nd,             # dzvel
    ctypes.c_int,    # nvel
    _nd,             # mx (in/out)
    _nd,             # my (in/out)
    _nd,             # mz (in/out)
    ctypes.c_int,    # mode
    _nd,             # spoil
]
_lib.blochsimfz.restype = None


# ---------------------------------------------------------------------------
# High-level Python wrapper
# ---------------------------------------------------------------------------
def bloch(b1, gr, tp,
          t1=np.inf, t2=np.inf, df=0.0, dp=0.0, dv=0.0,
          mode=0, mx0=None, my0=None, mz0=None, spoil=None):
    """
    Bloch equation simulator (ctypes wrapper for blochsimfz).

    Parameters
    ----------
    b1 : array_like, shape (ntime,)
        Complex RF pulse (Hz). Real part = Bx, imaginary part = By.
    gr : array_like, shape (ntime,) or (ntime, 2) or (ntime, 3)
        Gradient waveform (Hz/units). Columns are x [, y [, z]].
    tp : float or array_like, shape (ntime,)
        Duration of each time step (s). A scalar is broadcast to ntime.
    t1 : float
        T1 relaxation time (s). Default is np.inf (no T1 relaxation).
    t2 : float
        T2 relaxation time (s). Default is np.inf (no T2 relaxation).
    df : array_like, shape (nfreq,)
        Off-resonance frequencies (Hz). Default is 0.0 (on-resonance).
    dp : array_like, shape (npos,) or (npos, 2) or (npos, 3)
        Spatial positions (units). Columns are x [, y [, z]].
        Default is 0.0 (single position at the origin).
    dv : array_like, shape (nvel,) or (nvel, 2) or (nvel, 3)
        Velocities (units/s). Columns are x [, y [, z]].
        Default is 0.0 (stationary).
    mode : int, optional
        0 — return magnetisation only at the endpoint (default).
        1 — return magnetisation at every time step.
    mx0, my0, mz0 : array_like, shape (nvel, nfreq, npos), optional
        Initial magnetisation components. Defaults to [0, 0, 1] everywhere.
    spoil : array_like, shape (ntime,), optional
        Per-step spoiling flags (1 = zero Mx/My before that step).
        Defaults to all zeros (no spoiling).

    Returns
    -------
    mx, my, mz : ndarray
        Magnetisation components.
        Shape is (nvel, nfreq, npos) for mode=0, or
        (nvel, nfreq, npos, ntime) for mode=1.
        Singleton dimensions are squeezed.
    """
    # --- b1 ---
    b1 = np.asarray(b1, dtype=complex).ravel()
    ntime = len(b1)
    b1r = np.ascontiguousarray(b1.real, dtype=np.float64)
    b1i = np.ascontiguousarray(b1.imag, dtype=np.float64)

    # --- gradients ---
    gr = np.asarray(gr, dtype=np.float64)
    if gr.ndim == 1:
        gr = gr.reshape(-1, 1)
    ncols = gr.shape[1]
    gx = np.ascontiguousarray(gr[:, 0])
    gy = np.ascontiguousarray(gr[:, 1] if ncols >= 2 else np.zeros(ntime))
    gz = np.ascontiguousarray(gr[:, 2] if ncols >= 3 else np.zeros(ntime))

    # --- time steps ---
    tp = np.asarray(tp, dtype=np.float64).ravel()
    if tp.size == 1:
        tp = np.full(ntime, tp[0])
    tp = np.ascontiguousarray(tp)

    # --- off-resonance frequencies ---
    df = np.ascontiguousarray(np.asarray(df, dtype=np.float64).ravel())
    nfreq = len(df)

    # --- positions ---
    dp = np.atleast_1d(np.asarray(dp, dtype=np.float64))
    if dp.ndim == 1:
        dp = dp.reshape(-1, 1)
    npos = dp.shape[0]
    ncols = dp.shape[1]
    dx = np.ascontiguousarray(dp[:, 0])
    dy = np.ascontiguousarray(dp[:, 1] if ncols >= 2 else np.zeros(npos))
    dz = np.ascontiguousarray(dp[:, 2] if ncols >= 3 else np.zeros(npos))

    # --- velocities ---
    dv = np.atleast_1d(np.asarray(dv, dtype=np.float64))
    if dv.ndim == 1:
        dv = dv.reshape(-1, 1)
    nvel = dv.shape[0]
    ncols = dv.shape[1]
    dvx = np.ascontiguousarray(dv[:, 0])
    dvy = np.ascontiguousarray(dv[:, 1] if ncols >= 2 else np.zeros(nvel))
    dvz = np.ascontiguousarray(dv[:, 2] if ncols >= 3 else np.zeros(nvel))

    # --- spoil ---
    if spoil is None:
        spoil = np.zeros(ntime, dtype=np.float64)
    spoil = np.ascontiguousarray(np.asarray(spoil, dtype=np.float64).ravel())

    # --- allocate output arrays ---
    # C inner loop order: vel → freq → pos, with ntout values per combination.
    ntout = ntime if mode == 1 else 1
    total = nvel * nfreq * npos * ntout
    mx = np.zeros(total, dtype=np.float64)
    my = np.zeros(total, dtype=np.float64)
    mz = np.zeros(total, dtype=np.float64)

    # Set initial magnetisation.
    # blochsim() reads *mx, *my, *mz before the time loop, so write the
    # starting value into element [idx * ntout] of each flat output array.
    if mx0 is not None or my0 is not None or mz0 is not None:
        _mx0 = np.asarray(mx0 if mx0 is not None else np.zeros((nvel, nfreq, npos)), dtype=np.float64).ravel()
        _my0 = np.asarray(my0 if my0 is not None else np.zeros((nvel, nfreq, npos)), dtype=np.float64).ravel()
        _mz0 = np.asarray(mz0 if mz0 is not None else np.ones( (nvel, nfreq, npos)), dtype=np.float64).ravel()
        for idx in range(nvel * nfreq * npos):
            mx[idx * ntout] = _mx0[idx]
            my[idx * ntout] = _my0[idx]
            mz[idx * ntout] = _mz0[idx]
    else:
        for idx in range(nvel * nfreq * npos):
            mz[idx * ntout] = 1.0

    # --- call C ---
    _lib.blochsimfz(b1r, b1i, gx, gy, gz, tp, ntime,
                    t1, t2, df, nfreq,
                    dx, dy, dz, npos,
                    dvx, dvy, dvz, nvel,
                    mx, my, mz, mode, spoil)

    # --- reshape to (nvel, nfreq, npos, ntout) and squeeze singletons ---
    shape = (nvel, nfreq, npos, ntout)
    mx = mx.reshape(shape).squeeze()
    my = my.reshape(shape).squeeze()
    mz = mz.reshape(shape).squeeze()

    return mx, my, mz
