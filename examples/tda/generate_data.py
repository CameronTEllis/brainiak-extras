import logging
import numpy as np
from brainiak.utils import fmrisim as sim
from brainiak_extras.tda.rips import rips_filtration as rips

logger = logging.getLogger(__name__)

# Inputs for generate_signal
dimensions = np.array([64, 64, 36])  # What is the size of the brain
feature_size = [5, 9, 10, 6]
feature_type = ['cube', 'loop', 'cavity', 'sphere']
coordinates = np.array(
    [[32, 32, 18], [20, 32, 18], [32, 20, 18], [32, 32, 12]])
signal_magnitude = [1, -1, 1, -1]


# Inputs for generate_stimfunction
onsets = [10, 30, 50, 70, 90]
event_durations = [6]
tr_duration = 2
duration = 100

# TDA parameters
threshold = 99 # What percentile voxels will be included
max_dim = 2  # What is the maximum h level of your analysis
max_scale = float('inf')  # What is the search space of the parameters

# Generate a volume representing the location and quality of the signal
volume_static = sim.generate_signal(dimensions=dimensions,
                                    feature_coordinates=coordinates,
                                    feature_type=feature_type,
                                    feature_size=feature_size,
                                    signal_magnitude=signal_magnitude,
                                    )

# Create the time course for the signal to be generated
stimfunction = sim.generate_stimfunction(onsets=onsets,
                                         event_durations=event_durations,
                                         total_time=duration,
                                         )

# Convolve the HRF with the stimulus sequence
signal_function = sim.double_gamma_hrf(stimfunction=stimfunction,
                                       tr_duration=tr_duration,
                                       )

# Multiply the HRF timecourse with the signal
signal = sim.apply_signal(signal_function=signal_function,
                          volume_static=volume_static,
                          )

# Generate the mask of the signal
mask = sim.mask_brain(signal)

# Mask the signal to the shape of a brain (attenuates signal according to grey
# matter likelihood)
signal *= mask

# Create the noise volumes (using the default parameters
noise = sim.generate_noise(dimensions=dimensions,
                           stimfunction=stimfunction,
                           tr_duration=tr_duration,
                           mask=mask,
                           )

# Combine the signal and the noise
brain = signal + noise

# Reshape brain to be voxel by time
brain_mat = brain.reshape([dimensions[0] * dimensions[1] * dimensions[2],
                           brain.shape[3]])

# Which voxels are most variable
var = np.var(brain_mat, 1)
threshold_var = np.percentile(var, threshold)

# Select only those voxels that are most variable
brain_mat = brain_mat[var >= threshold_var,]

# The correlate each voxel with every other voxel. One minus the absolute
# value of this gives you a distance matrix
distance_matrix = 1 - np.abs(np.corrcoef(brain_mat))

# Perform the rips filtration
pairs = rips(max_dim=max_dim,
             max_scale=max_scale,
             dist_mat=distance_matrix,
             )

for triplet in pairs:
    print("Birth: %s Death: %s Dimension: %s" % triplet)