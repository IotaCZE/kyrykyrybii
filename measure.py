import numpy as np

from scipy.ndimage import distance_transform_edt


mask_old = np.load('mask_old.npy')
mask_new = np.load('mask_new.npy')

# Outside distances: distance from each point outside to the contour
outside_mask = np.ones_like(mask_old)
outside_mask[mask_old] = 0
dist_o, ind_o = distance_transform_edt(outside_mask, return_indices=True)
dist_o[mask_new == 0] = 0
outside_dist = np.max(dist_o)

max_pos_o = np.unravel_index(np.argmax(dist_o), dist_o.shape) 
nearest_y, nearest_x = ind_o[0][max_pos_o], ind_o[1][max_pos_o]
dy_o = nearest_y - max_pos_o[0]
dx_o = nearest_x - max_pos_o[1]
angle_o = np.arctan2(dy_o, dx_o)



# Inside distances: distance from each point inside to the contour
inside_mask = np.ones_like(mask_new)
inside_mask[mask_new] = 0
dist_i, ind_i = distance_transform_edt(inside_mask, return_indices=True)
dist_i[mask_old == 0] = 0
inside_dist = np.max(dist_i)

max_pos_i = np.unravel_index(np.argmax(dist_i), dist_i.shape)
nearest_y, nearest_x = ind_i[0][max_pos_i], ind_i[1][max_pos_i]
dy_i = nearest_y - max_pos_i[0]
dx_i = nearest_x - max_pos_i[1]
angle_i = np.arctan2(dy_i, dx_i)



print("Outside max distance: ", outside_dist)
print("Outside max position: ", max_pos_o)
print("Outside angle: ", angle_o)

print("Inside max distance: ", inside_dist)
print("Inside max position: ", max_pos_i)
print("Inside angle: ", angle_i)




