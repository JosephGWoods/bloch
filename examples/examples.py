import numpy as np
from bloch import bloch
from bloch_plot import animate_bloch, plot_bloch

# Simple hard pulse example: 90° flip at on-resonance, single position
dur    = 1e-3                             # pulse duration (s)
dt     = 10e-6                            # 10 us per time step
ntime  = int(dur / dt)                    # number of time steps
flip   = np.pi / 2                        # 90° flip angle
b1_amp = flip / (2 * np.pi * ntime * dt)  # B1+ amplitude (Hz)
b1     = b1_amp * np.ones(ntime)          # B1+ waveform (Hz)
gr     = np.zeros(ntime)                  # gradient waveform (Hz/m)

mx, my, mz = bloch(b1, gr, dt, mode=1)  # Run the Bloch simulation
animate_bloch(mx, my, mz, dt=dt,        # Create an animation of the magnetization vector
              filepath='examples/media/bloch_hard_90deg.mp4',
              title='90° hard pulse',
              unit_sphere=False)

# Slice-selective sinc pulse: animated profile and static endpoint
dur    = 2e-3                                          # pulse duration (s)
dt     = 10e-6                                         # 10 us per time step
ntime  = int(dur / dt)                                 # number of time steps
flip   = np.pi / 2                                     # 90° flip angle
slthk  = 5e-3                                          # slice thickness (m)
TBWP   = 8                                             # time-bandwidth product
b1     = np.sinc(np.linspace(-TBWP/2, TBWP/2, ntime))  # normalized sinc pulse
b1    *= flip / (2 * np.pi * np.sum(b1) * dt)          # B1+ waveform (Hz)
gr     = np.ones(ntime) * TBWP / dur / slthk           # gradient waveform (Hz/m)

dp = np.linspace(-slthk, slthk, 201)  # positions (m)

# Animated profile
mx, my, mz = bloch(b1, gr, dt, dp=dp, mode=1)  # run the Bloch simulation
animate_bloch(mx, my, mz, dt=dt, dp=dp*1e3,    # create an animation of the magnetization profile across positions
              xlabel='Position (mm)', title='90° Sinc pulse',
              filepath='examples/media/bloch_sinc_90deg.mp4')

# Static endpoint profile
mx, my, mz = bloch(b1, gr, dt, dp=dp, mode=0)  # run the Bloch simulation to get the final profile
plot_bloch(mx, my, mz, dp=dp*1e3,
           xlabel='Position (mm)', title='90° Sinc pulse',
           filepath='examples/media/bloch_sinc_90deg.png')


