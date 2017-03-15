# Generate data for TDA.
#
# This creates neural data which conforms to a circular graph structure.
# Nodes (for instance, different stimuli or tasks) are organized into
# a graph with each node equally spaced on the circumference of a circle.
# Each node is activatedone after the other, forming a looping
# sequence. This is
#  then reversed half way through the experiment.
# This graph structure is represented either in the pattern of activation in
#  the spatial data (either in a connected component or distributed across
# regions of the brain), or in the pattern of activation of a group of voxels
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
dimensions = np.array([24, 24, 18])  # What is the size of the brain
feature_size = [2]  # What is the size of each feature
feature_type = ['cube']
different_feature_loci = 2  # How many locations is the signal spread across?
signal_magnitude = [1]

# Graph structure
is_spatial = 1  # Does the similarity manifest as a spatial or temporal pattern
nodes = 12  # How many nodes are to be created

# What are the distances on the graph?
node_strength = 0.5  # What is the radius of the circle nodes

# Analysis decisions
if is_spatial:
    hrf_lag = 3  # How many TRs is the HRF lag for averaging?
else:
    hrf_lag = 0  # Remove this if you are using the timecourse

# How much noise do you want?
noise_dict = {}
noise_dict['overall'] = 0.1

# Inputs for generate_stimfunction
tr_duration = 2
event_durations = [feature_size[0]**3 * tr_duration]
events_per_node = 75  # How many repetitions of each node are there
time_between_nodes = 12
trial_dur = (event_durations[0] + time_between_nodes)  # How long is each trial
duration = trial_dur * events_per_node * nodes

# TDA parameters
threshold = 90  # What percentile voxels will be included
max_dim = 1  # What is the maximum h level of your analysis
max_scale = float('inf')  # What is the search space of the parameters

# Preset brain size
if is_spatial:
    node_brain = np.zeros([dimensions[0], dimensions[1], dimensions[2],
                           nodes])  # Preset
else:
    node_brain = np.zeros([dimensions[0], dimensions[1], dimensions[2],
                           int(event_durations[0] / tr_duration), nodes])


def generate_circle(node_strength):

    # The graph is a circle
    signal_coords = np.zeros(shape=[2, nodes])
    for node_counter in list(range(0, nodes)):
        # Where in the similarity space is this node
        radians = (2 * np.pi / nodes) * node_counter

        # Insert these coordinates into the signal
        signal_coords[0, node_counter] = node_strength * np.sin(radians)
        signal_coords[1, node_counter] = node_strength * np.cos(radians)

    # Return the signal
    return signal_coords


def persistence_diagram(barcode):
    # Print a persistence diagram
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
            color[feature_counter] = 'green'
        elif dimension == 2:
            color[feature_counter] = 'black'
        elif dimension == 3:
            color[feature_counter] = 'blue'

    # Rename the last distance, not as inf
    death[death.index(float('inf'))] = -1
    death[death.index(-1)] = np.max(death)

    # Plot the points
    plt.scatter(birth, death, c=color)
    plt.plot([np.min(death+birth), np.max(death+birth)], [np.min(death+birth),
                                                          np.max(death+birth)])


# Generate the graph structure
signal_coords = generate_circle(node_strength)

# Embed the coordinates in n dimensional space
if is_spatial:
    vector_size = (feature_size[0] ** 3) * different_feature_loci
else:
    vector_size = (event_durations[0] / tr_duration) * different_feature_loci

# Find normal vectors
vector_1 = np.random.randn(vector_size)
vector_2 = np.random.randn(vector_size)

# Norm the vector
vector_1 = vector_1 / np.linalg.norm(vector_1)
vector_2 = vector_2 / np.linalg.norm(vector_2)

# Combine vectors
ortho_normal_matrix = np.vstack((vector_1, vector_2))


# Orthogonalize the vectors using the Gram-Schmidt method
def gramschmidt(input):
    output = input[0:1, :].copy()
    for i in range(1, input.shape[0]):
        proj = np.diag(
            (input[i, :].dot(output.T) / np.linalg.norm(output, axis=1) **
             2).flat).dot(output)
        output = np.vstack((output, input[i, :] - proj.sum(0)))
    output = np.diag(1 / np.linalg.norm(output, axis=1)).dot(output)
    return output.T

ortho_normal_matrix = gramschmidt(ortho_normal_matrix)

# Re store the coordinates in the new dimensional space
signal_coords = np.dot(ortho_normal_matrix, signal_coords)

# Until you find a set of indices that haven't been used, iterate
# over them
x_idxs = [0] * different_feature_loci
y_idxs = [0] * different_feature_loci
z_idxs = [0] * different_feature_loci
used_idx = []
for loci_counter in list(range(0, different_feature_loci)):
    proposed_idxs = []
    while len(proposed_idxs) == 0:

        # Identify a location of the feature
        feature_loci = np.array([[np.random.randint(int(dimensions[0] * .3),
                                                    int(dimensions[0] * .6)),
                                  np.random.randint(int(dimensions[1] * .3),
                                                    int(dimensions[1] * .6)),
                                  np.random.randint(int(dimensions[2] * .3),
                                                    int(dimensions[2] * .6)),
                                  ]])

        # Pull out the idxs for where to insert the data
        x_idx, y_idx, z_idx = sim._insert_idxs(np.asarray(feature_loci)[0],
                                               feature_size[0],
                                               dimensions)

        # Convert 3d indices into 1d index
        for x_counter in list(range(x_idx[0], x_idx[1])):
            for y_counter in list(range(y_idx[0], y_idx[1])):
                for z_counter in list(range(z_idx[0], z_idx[1])):
                    idx = np.ravel_multi_index([x_counter,
                                                y_counter,
                                                z_counter],
                                               (dimensions[0],
                                                dimensions[1],
                                                dimensions[2]))
                    proposed_idxs = np.append(proposed_idxs, idx)

        # Check if an idx has been used before
        idx_used = 0
        for idx, proposed_idx in enumerate(proposed_idxs):
            if proposed_idx in used_idx:
                idx_used = 1

        # If no idxs overlapped then add this to the list
        if idx_used == 0:
            used_idx = np.append(used_idx, proposed_idxs)
            x_idxs[loci_counter] = x_idx
            y_idxs[loci_counter] = y_idx
            z_idxs[loci_counter] = z_idx
        else:
            proposed_idxs = []

# Create the data, each node at a time
for node_counter in list(range(0, nodes)):

    # Preset the signal
    signal_pattern = np.ones(feature_size[0] ** 3 * different_feature_loci)

    # Take the coordinates from the signal template
    for coord_counter in list(range(0, signal_coords.shape[0])):
        signal_pattern[coord_counter] = signal_coords[coord_counter,
                                                      node_counter]

    # Preset value
    volume = np.zeros(dimensions)

    # Loop through all of the events (both forwards and backwards)
    first_half_dur = trial_dur * int(events_per_node / 2) * nodes
    onsets = list(range(trial_dur * node_counter, first_half_dur,
                        trial_dur * nodes))

    # Backwards list
    onsets = onsets + list(
        range(first_half_dur + (trial_dur * (nodes - node_counter - 1)),
              trial_dur * events_per_node * nodes, trial_dur * nodes))

    # Create the time course for the signal to be generated
    stimfunc = sim.generate_stimfunction(onsets=onsets,
                                         event_durations=event_durations,
                                         total_time=duration,
                                         temporal_resolution=1 / tr_duration,
                                         )

    # Input the signal into the specified locations in the brain
    for loci_counter in list(range(0, different_feature_loci)):

        # Pull out the x, y z indices
        x_idx = x_idxs[loci_counter]
        y_idx = y_idxs[loci_counter]
        z_idx = z_idxs[loci_counter]

        # What is the signal for this feature location?
        start_idx = loci_counter * feature_size[0] ** 3
        end_idx = (loci_counter + 1) * feature_size[0] ** 3
        loci_signal = signal_pattern[start_idx:end_idx]

        # Does the similarity pattern exist across voxels or time points?
        if is_spatial == 1:

            # Insert the signal into the Volume
            volume[x_idx[0]:x_idx[1], y_idx[0]:y_idx[1], z_idx[0]:z_idx[1]] = \
                loci_signal.reshape([feature_size[0], feature_size[0],
                                     feature_size[0]])

        else:

            # Set the values of this cube to 1
            volume[x_idx[0]:x_idx[1], y_idx[0]:y_idx[1], z_idx[0]:z_idx[1]] = 1

            # Make the signal variation in the voxel activity over time
            tr_counter = 0
            while tr_counter < len(stimfunc):

                # Replace the stimulus function weights with the signal
                if stimfunc[tr_counter] == 1:
                    end_idx = int(tr_counter + event_durations[0] /
                                  tr_duration)
                    stimfunc[tr_counter:end_idx] = loci_signal
                    tr_counter += int(event_durations[0] / tr_duration)
                else:
                    tr_counter += 1

    # Convolve the HRF with the stimulus sequence if your signal isn't changes
    if is_spatial:
        signal_func = sim.double_gamma_hrf(stimfunction=stimfunc,
                                           tr_duration=tr_duration,
                                           temporal_resolution=1 / tr_duration,
                                           )
    else:
        signal_func = stimfunc

    # Multiply the HRF timecourse with the signal
    signal = sim.apply_signal(signal_function=signal_func,
                              volume_static=volume,
                              )

    # Generate the mask of the signal
    mask = sim.mask_brain(signal) > 0

    # Mask the signal to the shape of a brain (attenuates signal according
    # to grey matter likelihood)
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

    # Do something different for different analysis types
    if is_spatial:
        # Average the timepoints to make a single brain representing each node
        node_brain[:,:,:,node_counter] = np.mean(brain[:,:,:,node_trs],3)
    else:
        # Average all trials to make an average timecourse of activity

        # Overlay every trial
        tr_counter = 0
        while tr_counter < len(node_trs):
            trial_trs = node_trs[tr_counter:int(tr_counter +
                                                event_durations[0] /
                                                tr_duration)]

            node_brain[:, :, :, :, node_counter] += brain[:, :, :, trial_trs]

            tr_counter += len(trial_trs)

        # Take the average of all the timepoints
        node_brain[:, :, :, :, node_counter] = node_brain[:, :, :, :,
                                               node_counter] / len(
            node_trs) / (event_durations[0] / tr_duration)

# Set up and threshold the node by voxel matrix
if is_spatial:
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
else:
    # If it is in time then reshape to be a 3d volume of voxel * time * node
    node_mat = node_brain.reshape([dimensions[0] * dimensions[1] *
                                   dimensions[2],
                                   int(event_durations[0] / tr_duration),
                                   nodes])

    # Which voxels are most variable, averaging across nodes
    var = np.var(np.mean(node_mat,2), 1)

    threshold_var = np.percentile(var, threshold)

    # Select only those voxels that are most variable
    node_mat = node_mat[var > threshold_var,:,:]

    # Reshape so that each timepoint is concatenated
    node_mat = node_mat.reshape([node_mat.shape[0] * node_mat.shape[1],
    nodes])

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
persistence_diagram(barcode)