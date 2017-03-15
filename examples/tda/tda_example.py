# Generate data for TDA.
#
# This creates neural data which conforms to a circular graph structure.
# Nodes (for instance, different stimuli) are organized into
# a graph with each node equally spaced on the circumference of a circle.
# Each event occurs one after the other, forming a looping sequence. This is
#  then reversed half way through the experiment.
# This graph structure is represented either in the pattern of activation in
#  the spatial data, or in the pattern of activation of a group of voxels
# across time.

import logging
from brainiak.utils import fmrisim as sim
from brainiak_extras.tda.rips import rips_filtration as rips
import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance as sp_distance
from sklearn import manifold

logger = logging.getLogger(__name__)

# Inputs for generate_signal
dimensions = np.array([24, 24, 12])  # What is the size of the brain
feature_size = [2]
feature_type = ['cube']
coordinates = np.array(
    [[12, 12, 4]])
signal_magnitude = [1]

nodes = 15  # How many nodes are to be created
node_strength = 3 # What is the radius of the circle nodes

# Does the similarity manifest as a spatial pattern or as a temporal pattern
is_spatial = 1
hrf_lag = 3  # How many TRs is the HRF lag (what TRs are you averaging for
# an event?

# How much noise do you want?
noise_dict = {}
noise_dict['overall'] = 0

# Inputs for generate_stimfunction
event_durations = [2]#[feature_size[0]**3]
tr_duration = 2
events_per_node = 6  # How many repetitions of each node are there
time_between_nodes = 2#12
trial_dur = (event_durations[0] + time_between_nodes)  # How long is each trial
duration = trial_dur * events_per_node * nodes

# TDA parameters
threshold = 90  # What percentile voxels will be included
max_dim = 1  # What is the maximum h level of your analysis
max_scale = float('inf')  # What is the search space of the parameters

# Generate a volume representing the location and quality of the signal
volume_template = sim.generate_signal(dimensions=dimensions,
                                      feature_coordinates=coordinates,
                                      feature_type=feature_type,
                                      feature_size=feature_size,
                                      signal_magnitude=signal_magnitude,
                                      )

# Iterate through the nodes, creating a new volume each time
node_brain = np.zeros([dimensions[0], dimensions[1], dimensions[2],
                       nodes])  # Preset
for node_counter in list(range(0, nodes)):

    # Where in the similarity space is this node
    radians = (2 * np.pi / nodes) * node_counter
    x_coord = node_strength * np.sin(radians)
    y_coord = node_strength * np.cos(radians)

    # Preset value
    volume = volume_template

    # Does the similarity pattern exist across voxels or across time points?
    if is_spatial == 1:
        volume[coordinates[0][0], coordinates[0][1], coordinates[
            0][2]] = x_coord
        volume[coordinates[0][0], coordinates[0][1], coordinates[0][2]
                        + 1] = y_coord

        # Loop through all of the events (both forwards and backwards)

        first_half_dur = trial_dur * int(events_per_node / 2) * nodes
        onsets = list(range(trial_dur * node_counter, first_half_dur,
                            trial_dur * nodes))

        # Backwards list
        onsets = onsets + list(range(first_half_dur + (trial_dur * (nodes - node_counter - 1)), trial_dur * events_per_node * nodes, trial_dur * nodes))

        # Create the time course for the signal to be generated
        stimfunc = sim.generate_stimfunction(onsets=onsets,
                                             event_durations=event_durations,
                                             total_time=duration,
                                             temporal_resolution=1 /
                                                                 tr_duration,
                                             )

        # Convolve the HRF with the stimulus sequence
        signal_func = sim.double_gamma_hrf(stimfunction=stimfunc,
                                           tr_duration=tr_duration,
                                           temporal_resolution=1 / tr_duration,
                                           )

        # Multiply the HRF timecourse with the signal
        signal = sim.apply_signal(signal_function=signal_func,
                                  volume_static=volume,
                                  )

        # Generate the mask of the signal
        mask = sim.mask_brain(signal) > 0

        # Mask the signal to the shape of a brain (attenuates signal according to grey
        # matter likelihood)
        signal *= mask

        # Create the noise volumes (using the default parameters)
        noise = sim.generate_noise(dimensions=dimensions,
                                   stimfunction_tr=stimfunc,
                                   tr_duration=tr_duration,
                                   mask=mask,
                                   noise_dict=noise_dict,
                                   )

        # Combine the signal and the noise
        brain = signal + noise

        # Find all the timepoints corresponding to this node
        node_trs = []
        for onset_counter in list(range(0, len(onsets))):

            # What are the relevant TRs for this event
            tr_start = np.round(onsets[onset_counter] / tr_duration) + hrf_lag
            tr_end = np.round((onsets[onset_counter] + event_durations[0]) / \
                     tr_duration) + hrf_lag

            if tr_start < brain.shape[3] and tr_end < brain.shape[3]:
                node_trs += list(range(int(tr_start), int(tr_end)))

        # Average the timepoints to make a single brain representing each node
        node_brain[:,:,:,node_counter] = np.mean(brain[:,:,:,node_trs],3)


# Plot the representation of the data
plt.scatter(node_brain[coordinates[0][0], coordinates[0][1], coordinates[
            0][2], :], node_brain[coordinates[0][0], coordinates[0][1],
                       coordinates[0][2] + 1, :])

# Reshape the average nodes to be voxel by time
node_mat = node_brain.reshape([dimensions[0] * dimensions[1] * dimensions[2],
                                nodes])

# Which voxels are most variable
var = np.var(node_mat, 1)
threshold_var = np.percentile(var, threshold)

# Select only those voxels that are most variable
node_mat = node_mat[var > threshold_var,]

# Take the transpose so that it is node by pattern across voxels
node_mat = np.transpose(node_mat)

# Find the distance matrix of the nodes across the pattern of voxels
distance_matrix = sp_distance.squareform(sp_distance.pdist(node_mat))

# Create an MDS plot
plt.figure()
mds = manifold.MDS(n_components=2) # Fit the mds object
coords = mds.fit(distance_matrix).embedding_ # Find the mds coordinates
plt.scatter(coords[:,0], coords[:,1]) # Plot the MDS

# Perform the rips filtration to get a barcode
barcode = rips(max_dim=max_dim,
               max_scale=max_scale,
               dist_mat=distance_matrix,
               )

# Print the barcodes
for triplet in barcode:
    print("Birth: %s Death: %s Dimension: %s" % triplet)

# Print a persistence diagram
plt.figure()
birth = [0] * len(barcode)
death = [0] * len(barcode)
color = [0] * len(barcode)
for feature_counter in list(range(0, len(barcode))):
    birth[feature_counter] = barcode[feature_counter][0]
    death[feature_counter] = barcode[feature_counter][1]
    dimension = barcode[feature_counter][2]

    # Assign the color
    if dimension == 0:
        color[feature_counter] = 'red'
    elif dimension == 1:
        color[feature_counter] = 'orange'
    elif dimension == 2:
        color[feature_counter] = 'yellow'
    elif dimension == 3:
        color[feature_counter] = 'green'

# Rename the last distance, not as inf
death[death.index(float('inf'))] = -1
death[death.index(-1)] = np.max(death)

# Plot the points
plt.scatter(birth, death, c=color)
plt.plot([np.min(birth), np.max(birth)], [np.min(death), np.max(death)])
