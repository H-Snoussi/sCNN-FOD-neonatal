import nibabel as nib
import numpy as np
import os
import random
from scipy.spatial import distance_matrix

# def farthest_point_sampling(bvecs, num_selected):
#     """
#     Selects a subset of gradient directions that maximizes angular coverage using 
#     a farthest point sampling approach.
#     """
#     bvecs = bvecs / np.linalg.norm(bvecs, axis=0, keepdims=True)  # Normalize

#     # Start with a randomly chosen first point
#     selected_indices = [0]
#     distances = np.full(bvecs.shape[1], np.inf)

#     for _ in range(num_selected - 1):
#         # Compute distances of each point to the closest selected point
#         dists = np.linalg.norm(bvecs[:, selected_indices].T[:, None] - bvecs.T, axis=2)
#         min_dists = np.min(dists, axis=0)
#         distances = np.minimum(distances, min_dists)

#         # Pick the farthest point
#         next_index = np.argmax(distances)
#         selected_indices.append(next_index)

#     return np.array(selected_indices)

# def electrostatic_repulsion_selection(bvecs, num_selected, iterations=100, step_size=0.1):
#     """ Selects a subset of diffusion directions using electrostatic repulsion. """

#     # Normalize bvecs to unit vectors
#     bvecs = bvecs / np.linalg.norm(bvecs, axis=0, keepdims=True)

#     # Initialize with a random subset
#     selected_indices = np.random.choice(bvecs.shape[1], num_selected, replace=False)
#     selected_bvecs = bvecs[:, selected_indices]

#     for _ in range(iterations):
#         forces = np.zeros_like(selected_bvecs)

#         # Compute electrostatic repulsion forces
#         for i in range(num_selected):
#             diffs = selected_bvecs[:, i].reshape(3, 1) - selected_bvecs
#             norms = np.linalg.norm(diffs, axis=0, keepdims=True) + 1e-8
#             forces[:, i] += np.sum(diffs / norms**3, axis=1)  # Coulomb force

#         # Update positions
#         selected_bvecs += step_size * forces
#         selected_bvecs = selected_bvecs / np.linalg.norm(selected_bvecs, axis=0, keepdims=True)  # Re-normalize

#     # Find closest original gradients to final repelled points
#     final_indices = []
#     for vec in selected_bvecs.T:
#         distances = np.linalg.norm(bvecs.T - vec, axis=1)
#         final_indices.append(np.argmin(distances))

#     return np.array(final_indices)

# # Load bvals and bvecs
# bvals = np.loadtxt("/local/projects/scnn-investigation_haykel/protocols/sub-07/dwi.bval")
# bvecs = np.loadtxt("/local/projects/scnn-investigation_haykel/protocols/sub-07/dwi.bvec")

# # Define shells
# shells = {400: 64, 1000: 88, 2600: 128}  # Excluding b=0 volumes
# selected_indices = []

# # Process each shell separately
# for bval, total_dirs in shells.items():
#     shell_idx = np.where(np.isclose(bvals, bval, atol=50))[0]
#     num_selected = total_dirs // 2  # Select half of the directions

#     # shell_selected = electrostatic_repulsion_selection(bvecs[:, shell_idx], num_selected)
#     shell_selected = farthest_point_sampling(bvecs[:, shell_idx], num_selected)

#     selected_indices.extend(shell_idx[shell_selected])  # Map to original indices

# selected_indices = np.array(selected_indices)

# # Extract corresponding bvals and bvecs
# selected_bvals = bvals[selected_indices]
# selected_bvecs = bvecs[:, selected_indices]

# # Save for later use
# np.savetxt("/local/projects/scnn-investigation_haykel/protocols/sub-07/selected_indices.txt", selected_indices, fmt="%d")
# np.savetxt("/local/projects/scnn-investigation_haykel/protocols/sub-07/selected_bvals.txt", selected_bvals, fmt="%d")
# np.savetxt("/local/projects/scnn-investigation_haykel/protocols/sub-07/selected_bvecs.txt", selected_bvecs, fmt="%.6f")



import numpy as np
import os

# Load bvals and bvecs
bvals = np.loadtxt("/local/projects/scnn-investigation_haykel/protocols/sub-07/dwi.bval")
bvecs = np.loadtxt("/local/projects/scnn-investigation_haykel/protocols/sub-07/dwi.bvec")

# Define shells
shells = {400: 64, 1000: 88, 2600: 128}  # Excluding b=0 volumes
selected_indices = []

# Process each shell separately
for bval, total_dirs in shells.items():
    shell_idx = np.where(np.isclose(bvals, bval, atol=50))[0]
    # num_selected = total_dirs // 2  # Select the first half
    num_selected = int(total_dirs * 0.3)  # Select 90% of directions

    shell_selected = shell_idx[:num_selected]  # Select first N indices
    selected_indices.extend(shell_selected)  # Map to original indices

selected_indices = np.array(selected_indices)

# Extract corresponding bvals and bvecs
selected_bvals = bvals[selected_indices]
selected_bvecs = bvecs[:, selected_indices]

# Save for later use
np.savetxt("/local/projects/scnn-investigation_haykel/protocols/sub-07/selected_indices.txt", selected_indices, fmt="%d")
np.savetxt("/local/projects/scnn-investigation_haykel/protocols/sub-07/selected_bvals.txt", selected_bvals, fmt="%d")
np.savetxt("/local/projects/scnn-investigation_haykel/protocols/sub-07/selected_bvecs.txt", selected_bvecs, fmt="%.6f")

print("Selection completed: First half of indices, bvals, and bvecs saved.")



# python scripts/select_half_directions.py
# python scripts/extract_half_dmri.py
# python scripts/prepare_sh_coeffs.py
# python scripts/inference_good2.py