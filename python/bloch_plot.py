"""
bloch_animation.py — Bloch sphere animation and profile plotting helpers.

Two public functions:

``animate_bloch``  — animates mode=1 output over time.
  - Single spin (no dp/df): rotating vector on the Bloch sphere.
  - Multi-position (dp given): Mx/My/Mz vs position, animated over time.
  - Multi-frequency (df given): Mx/My/Mz vs off-resonance, animated over time.

``plot_bloch`` — static plot of mode=0 output.
  - Single spin: vector on the Bloch sphere.
  - Multi-position (dp given): Mx/My/Mz vs position.
  - Multi-frequency (df given): Mx/My/Mz vs off-resonance.

Usage
-----
    from bloch import bloch
    from bloch_plot import animate_bloch, plot_bloch

    # Single spin — Bloch sphere animation
    mx, my, mz = bloch(b1, gr, dt, mode=1)
    animate_bloch(mx, my, mz, dt=dt, filepath='spin.mp4')

    # Slice profile animation
    mx, my, mz = bloch(b1, gr, dt, dp=dp, mode=1)
    animate_bloch(mx, my, mz, dt=dt, dp=dp, filepath='profile.mp4')

    # Static slice profile
    mx, my, mz = bloch(b1, gr, dt, dp=dp)
    plot_bloch(mx, my, mz, dp=dp, filepath='profile.png')
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _save_animation(ani, fig, filepath, fps, dpi):
    """Save *ani* to *filepath* and close the figure."""

    if filepath.endswith('.gif'):
        writer = PillowWriter(fps=fps)
    else:
        writer = FFMpegWriter(fps=fps, bitrate=-1)
    ani.save(filepath, writer=writer, dpi=dpi)
    print(f'Saved: {filepath}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Bloch sphere helpers
# ---------------------------------------------------------------------------

def _unit_sphere(ax, alpha=0.08):
    """Draw a translucent unit sphere with equator and two meridians."""
    u, v = np.mgrid[0:2*np.pi:60j, 0:np.pi:30j]
    ax.plot_surface(
        np.sin(v) * np.cos(u),
        np.sin(v) * np.sin(u),
        np.cos(v),
        color='steelblue', alpha=alpha, linewidth=0, zorder=0,
    )
    theta = np.linspace(0, 2*np.pi, 200)
    kw = dict(color='grey', lw=0.5, alpha=0.5)
    ax.plot(np.cos(theta), np.sin(theta), np.zeros_like(theta), **kw)
    ax.plot(np.cos(theta), np.zeros_like(theta), np.sin(theta), **kw)
    ax.plot(np.zeros_like(theta), np.cos(theta), np.sin(theta), **kw)


def _draw_sphere_axes(ax, lw=1.2, r = 1.1):
    """Draw labelled Cartesian axes through the sphere."""
    for xyz, label in zip([(r,0,0),(0,r,0),(0,0,r)], ['x','y','z']):
        ax.plot([0, xyz[0]], [0, xyz[1]], [0, xyz[2]],
                color='black', lw=lw, alpha=0.6)
        ax.text(xyz[0]*1.08, xyz[1]*1.08, xyz[2]*1.08, label,
                fontsize=11, ha='center', va='center')
    for xyz in [(-r,0,0),(0,-r,0),(0,0,-r)]:
        ax.plot([0, xyz[0]], [0, xyz[1]], [0, xyz[2]],
                color='black', lw=lw, alpha=0.2, linestyle='--')


def _make_sphere_axes(fig, title, r = 1.1, unit_sphere=True):
    ax = fig.add_subplot(111, projection='3d')
    if unit_sphere:
        _unit_sphere(ax)
    _draw_sphere_axes(ax, r=r)
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_zlim(-r, r)
    ax.set_box_aspect([1, 1, 1], zoom=1.6)
    ax.set_axis_off()
    if title:
        ax.set_title(title, pad=4)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.95)
    return ax


# ---------------------------------------------------------------------------
# Profile helpers
# ---------------------------------------------------------------------------

def _make_profile_axes(fig, xlabel, title, x):
    """Three vertically-stacked subplots sharing the x-axis."""
    axs = fig.subplots(3, 1, sharex=True)
    for ax, label, color in zip(axs, ['Mx', 'My', 'Mz'], ['C0', 'C1', 'C2']):
        ax.set_ylabel(label, color=color)
        ax.tick_params(axis='y', labelcolor=color)
        ax.set_ylim(-1.1, 1.1)
        ax.set_xlim(x[0], x[-1])
        ax.axhline(0, color='grey', lw=0.5, ls='--')
    axs[-1].set_xlabel(xlabel)
    if title:
        axs[0].set_title(title)
    # tight_layout is called by the caller after all artists are added
    return axs


# ---------------------------------------------------------------------------
# Public: animate_bloch
# ---------------------------------------------------------------------------

def animate_bloch(mx, my, mz, dt=None, dp=None, df=None,
                  xlabel=None, title=None, filepath=None,
                  fps=30, dpi=200,
                  trace=True, trace_len=None, unit_sphere=True):
    """
    Animate a magnetisation trajectory (mode=1 output of bloch()).

    Dispatches automatically based on which optional arguments are supplied:

    - *dp* or *df* with more than one element → line-plot animation of
      Mx/My/Mz vs position or frequency over time.
    - Neither → Bloch sphere animation (single isochromat only).

    Parameters
    ----------
    mx, my, mz : array_like
        mode=1 magnetisation output of ``bloch()``.
        Shape ``(ntime,)`` for a single spin, or ``(n, ntime)`` for n
        positions/frequencies.
    dt : float, optional
        Duration of each time step (s). Shown as elapsed time on the plot.
    dp : array_like, shape (n,), optional
        Spatial positions passed to ``bloch()``. Triggers profile mode.
    df : array_like, shape (n,), optional
        Off-resonance frequencies passed to ``bloch()``. Triggers profile mode.
    xlabel : str, optional
        Profile mode only. X-axis label. Defaults to 'Position' or
        'Off-resonance frequency (Hz)'.
    title : str, optional
        Plot title.
    filepath : str, optional
        Save path. Extension selects format: ``.mp4`` (FFmpeg) or ``.gif``
        (Pillow). Returns the ``FuncAnimation`` without saving if *None*.
    fps : int
        Frames per second. Default 30.
    dpi : int
        Resolution of saved file. Default 200.
    trace : bool
        Bloch sphere mode only. Show trail of vector tip. Default True.
    trace_len : int, optional
        Bloch sphere mode only. Max trail length (frames). Default: full.
    unit_sphere : bool
        Show a translucent unit sphere for reference. Default True.

    Returns
    -------
    ani : FuncAnimation
    """
    mx = np.asarray(mx, dtype=float)
    my = np.asarray(my, dtype=float)
    mz = np.asarray(mz, dtype=float)

    # Detect profile mode
    x, default_xlabel = None, None
    if dp is not None and np.asarray(dp).size > 1:
        x, default_xlabel = np.asarray(dp, dtype=float).ravel(), 'Position'
    elif df is not None and np.asarray(df).size > 1:
        x, default_xlabel = np.asarray(df, dtype=float).ravel(), 'Off-resonance frequency (Hz)'

    if x is not None:
        return _animate_profile(mx, my, mz, x, dt=dt,
                                xlabel=xlabel or default_xlabel,
                                filepath=filepath,
                                title=title, fps=fps, dpi=dpi)
    else:
        return _animate_sphere(mx.ravel(), my.ravel(), mz.ravel(), dt=dt,
                               title=title, filepath=filepath,
                               fps=fps, dpi=dpi,
                               trace=trace, trace_len=trace_len,
                               unit_sphere=unit_sphere)


# ---------------------------------------------------------------------------
# Public: plot_bloch
# ---------------------------------------------------------------------------

def plot_bloch(mx, my, mz, dp=None, df=None,
               xlabel=None, title=None, filepath=None,
               dpi=200, unit_sphere=True):
    """
    Static plot of magnetisation (mode=0 output of bloch()).

    Dispatches automatically:

    - *dp* or *df* with more than one element → 3-panel line plot of
      Mx/My/Mz vs position or frequency.
    - Neither → Bloch sphere with the magnetisation vector.

    Parameters
    ----------
    mx, my, mz : array_like
        mode=0 magnetisation output of ``bloch()``.
    dp : array_like, shape (n,), optional
        Spatial positions. Triggers profile mode.
    df : array_like, shape (n,), optional
        Off-resonance frequencies. Triggers profile mode.
    xlabel : str, optional
        X-axis label (profile mode). Defaults to 'Position' or
        'Off-resonance frequency (Hz)'.
    title : str, optional
        Figure title.
    filepath : str, optional
        Save path. Extension selects format: ``.png`` or ``.jpg``. If *None*, the figure is not saved.
    dpi : int
        Resolution of saved file. Default 200.
    unit_sphere : bool
        Show a translucent unit sphere for reference. Default True.

    Returns
    -------
    fig : matplotlib Figure
    """
    mx = np.asarray(mx, dtype=float).ravel()
    my = np.asarray(my, dtype=float).ravel()
    mz = np.asarray(mz, dtype=float).ravel()

    x, default_xlabel = None, None
    if dp is not None and np.asarray(dp).size > 1:
        x, default_xlabel = np.asarray(dp, dtype=float).ravel(), 'Position'
    elif df is not None and np.asarray(df).size > 1:
        x, default_xlabel = np.asarray(df, dtype=float).ravel(), 'Off-resonance frequency (Hz)'

    if x is not None:
        fig = _plot_profile(mx, my, mz, x,
                             xlabel=xlabel or default_xlabel,
                             title=title)
    else:
        fig = _plot_sphere_static(mx[0], my[0], mz[0], title=title, unit_sphere=unit_sphere)

    if filepath is not None:
        fig.savefig(filepath, dpi=dpi)
        print(f'Saved: {filepath}')
    
    return fig

# ---------------------------------------------------------------------------
# Internal: Bloch sphere animation
# ---------------------------------------------------------------------------

def _animate_sphere(mx, my, mz, dt,
                    title, filepath, fps, dpi,
                    trace, trace_len, unit_sphere=True):

    ntime = len(mx)
    if trace_len is None:
        trace_len = ntime

    fig = plt.figure(figsize=(5, 5))
    ax = _make_sphere_axes(fig, title, unit_sphere=unit_sphere)

    quiver_ref = [None]
    trace_line, = ax.plot([], [], [], color='tomato', lw=1.0, alpha=0.5)
    time_text = ax.text2D(0.02, 0.96, '', transform=ax.transAxes,
                          fontsize=9, va='top')

    def _update(frame):
        if quiver_ref[0] is not None:
            quiver_ref[0].remove()
        quiver_ref[0] = ax.quiver(
            0, 0, 0, mx[frame], my[frame], mz[frame],
            length=1.0, normalize=False,
            color='tomato', linewidth=2.0, arrow_length_ratio=0.12,
        )
        if trace:
            start = max(0, frame - trace_len + 1)
            trace_line.set_data_3d(mx[start:frame+1],
                                   my[start:frame+1],
                                   mz[start:frame+1])
        if dt is not None:
            time_text.set_text(f't = {(frame+0.5) * dt * 1e3:.2f} ms')
        return quiver_ref[0], trace_line, time_text

    ani = FuncAnimation(fig, _update, frames=ntime,
                        interval=1000/fps, blit=False)
    if filepath is not None:
        _save_animation(ani, fig, filepath, fps, dpi)
    return ani


# ---------------------------------------------------------------------------
# Internal: Bloch sphere static plot
# ---------------------------------------------------------------------------

def _plot_sphere_static(mx_val, my_val, mz_val, title, unit_sphere=True):

    fig = plt.figure(figsize=(5, 5))
    ax = _make_sphere_axes(fig, title, unit_sphere=unit_sphere)
    ax.quiver(0, 0, 0, mx_val, my_val, mz_val,
              length=1.0, normalize=False,
              color='tomato', linewidth=2.0, arrow_length_ratio=0.12)
    return fig


# ---------------------------------------------------------------------------
# Internal: profile animation
# ---------------------------------------------------------------------------

def _animate_profile(mx, my, mz, x, dt,
                     xlabel, filepath, title, fps, dpi):
    # Ensure 2D: (n, ntime)
    mx = np.atleast_2d(mx)
    my = np.atleast_2d(my)
    mz = np.atleast_2d(mz)
    ntime = mx.shape[-1]

    fig = plt.figure(figsize=(6, 5))
    axs = _make_profile_axes(fig, xlabel, title, x)
    lines = []
    for ax, data, color in zip(axs, [mx, my, mz], ['C0', 'C1', 'C2']):
        line, = ax.plot(x, data[:, 0], color=color, lw=1.5)
        lines.append(line)

    time_text = axs[0].text(0.98, 0.90, '', transform=axs[0].transAxes,
                             fontsize=9, ha='right', va='top')

    fig.tight_layout()

    def _update(frame):
        for line, data in zip(lines, [mx, my, mz]):
            line.set_ydata(data[:, frame])
        if dt is not None:
            time_text.set_text(f't = {(frame+0.5) * dt * 1e3:.2f} ms')
        return (*lines, time_text)

    ani = FuncAnimation(fig, _update, frames=ntime,
                        interval=1000/fps, blit=False)
    if filepath is not None:
        _save_animation(ani, fig, filepath, fps, dpi)
    return ani


# ---------------------------------------------------------------------------
# Internal: static profile plot
# ---------------------------------------------------------------------------

def _plot_profile(mx, my, mz, x, xlabel, title):
    fig = plt.figure(figsize=(6, 5))
    axs = _make_profile_axes(fig, xlabel, title, x)
    for ax, data, color in zip(axs, [mx, my, mz], ['C0', 'C1', 'C2']):
        ax.plot(x, data, color=color, lw=1.5)
    fig.tight_layout()
    return fig
