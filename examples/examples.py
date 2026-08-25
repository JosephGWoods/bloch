import os
import numpy as np
from bloch import bloch
from bloch_plot import animate_bloch, plot_bloch

MEDIA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'media')

# Simple hard pulse example: 90° flip at on-resonance, single position
dur    = 1e-3                             # pulse duration (s)
dt     = 10e-6                            # 10 us per time step
ntime  = round(dur / dt)                  # number of time steps
flip   = np.pi / 2                        # 90° flip angle
b1_amp = flip / (2 * np.pi * ntime * dt)  # B1+ amplitude (Hz)
b1     = b1_amp * np.ones(ntime)          # B1+ waveform (Hz)
g      = np.zeros(ntime)                  # gradient waveform (Hz/m)

# Run the Bloch simulation with full time course output (mode=1)
mx, my, mz = bloch(b1, g, dt, mode=1)

# Generate an animation of the magnetization vector
animate_bloch(mx, my, mz, dt=dt,
              filepath=os.path.join(MEDIA_DIR, 'bloch_hard_90deg.mp4'),
              title='90° hard pulse',
              unit_sphere=False)

# Slice-selective windowed sinc pulse: animated profile and static endpoint
dur    = 2e-3                                              # pulse duration (s)
dt     = 10e-6                                             # 10 us per time step
ntime  = round(dur / dt)                                   # number of time steps
flip   = np.pi / 2                                         # 90° flip angle
slthk  = 5e-3                                              # slice thickness (m)
TBWP   = 8                                                 # time-bandwidth product
t      = (np.arange(1, ntime + 1) - 0.5) * dt - (dur / 2)  # time points centered on zero
alpha = 0.5                                                # apodization factor (Hann window)
window = (1 - alpha) + alpha * np.cos(2 * np.pi * t / dur) # Hann window
b1     = np.multiply(window, np.sinc(t * TBWP / dur))      # normalized sinc pulse
b1    *= flip / (2 * np.pi * np.sum(b1) * dt)              # B1+ waveform (Hz)
b1     = np.concatenate([b1, np.zeros(ntime//2)])          # pad RF with zeros during gradient rewinder
gz     = np.concatenate(
            [np.ones(ntime)    *  TBWP / dur / slthk,      # slice-selective gradient waveform (Hz/m)
             np.ones(ntime//2) * -TBWP / dur / slthk]      # slice rewinding gradient waveform (Hz/m)
        )

# Define positions for the simulation
dp = np.linspace(-slthk, slthk, 201) # (m)

# Run the Bloch simulation with full time course output (mode=1)
mx, my, mz = bloch(b1, gz, dt, dp=dp, mode=1)

# Generate an animation of the magnetization profile across positions
animate_bloch(mx, my, mz, dt=dt, dp=dp*1e3,
              b1=b1, g=gz, b1_units='uT', g_units='mT/m', b1_mode='mag_phase',
              xlabel='Position (mm)', title='90° Sinc pulse',
              filepath=os.path.join(MEDIA_DIR, 'bloch_sinc_90deg.mp4'))

# Run the Bloch simulation with only the final magnetization profile (mode=0)
mx, my, mz = bloch(b1, gz, dt, dp=dp, mode=0)  

# Plot the final magnetization profile across positions
plot_bloch(mx, my, mz, dp=dp*1e3,
           b1=b1, g=gz, dt=dt, b1_units='uT', g_units='mT/m', b1_mode='mag_phase',
           xlabel='Position (mm)', title='90° Sinc pulse',
           filepath=os.path.join(MEDIA_DIR, 'bloch_sinc_90deg.png'))


# Note:
# mp4 files converted to gif using ffmpeg:
# ffmpeg -i bloch_sinc_90deg.mp4 -vf "fps=15,scale=1000:-1:flags=lanczos" bloch_sinc_90deg.gif
