
import numpy as np


def extrapolate_edge_pixel_mat_define(mask, do_ext_2_pix=False):
    '''Python version of the equivalent oaa_lib routine,
    translated by ChatGPT
    
    TODO while translating, we found out that y extrapolation
    has precedence over x (because sum_1pix_extra and sum_2pix_extra
    are modified in-place). Is this intentional?
    '''

    # Get the dimensions of the mask
    smask = mask.shape
    idx_mask = np.where(mask)
    float_mask = mask.astype(float)
    
    # Initialize matrices for extrapolated sums
    sum_1pix_extra = np.full(float_mask.shape, -1.0)
    sum_2pix_extra = np.full(float_mask.shape, -1.0)
    idx_mask_array = np.zeros_like(float_mask)
    idx_mask_array[idx_mask] = np.ravel_multi_index(idx_mask, smask)

    # Define 1-pixel extrapolated pixels out of the pupil mask
    find_1pix_extra = (np.roll(float_mask, 1, axis=0) + np.roll(float_mask, -1, axis=0) +
                       np.roll(float_mask, 1, axis=1) + np.roll(float_mask, -1, axis=1))
    find_1pix_extra *= (1.0 - float_mask)
    
    # Define 2-pixel extrapolated pixels out of the pupil mask
    find_2pix_extra = (np.roll(float_mask, 2, axis=0) + np.roll(float_mask, -2, axis=0) +
                       np.roll(float_mask, 2, axis=1) + np.roll(float_mask, -2, axis=1))
    find_2pix_extra *= (1.0 - float_mask)

    # Get indices of 1-pixel and 2-pixel extrapolated regions
    idx_1pix_extra = np.where(find_1pix_extra > 0)
    idx_2pix_extra = np.where((find_2pix_extra > 0) & (find_1pix_extra == 0))
    
    # Iterate over each index in idx_1pix_extra
    for idx in zip(*idx_1pix_extra):
        if np.sum(idx) * np.size(idx) != -1:
            ind = np.array(idx)
            test = -1

            # y+ direction
            if ind[1] + 1 < smask[1] - 1 and sum_1pix_extra[idx] == -1:
                if float_mask[ind[0], ind[1] + 1] > 0:
                    if ind[1] + 2 < smask[1] - 1 and float_mask[ind[0], ind[1] + 2] > 0:
                        sum_2pix_extra[idx] = idx_mask_array[ind[0], ind[1] + 2]
                        sum_1pix_extra[idx] = idx_mask_array[ind[0], ind[1] + 1]
                    else:
                        test = idx_mask_array[ind[0], ind[1] + 1]
            
            # y- direction
            if ind[1] - 1 > 0 and sum_1pix_extra[idx] == -1:
                if float_mask[ind[0], ind[1] - 1] > 0:
                    if ind[1] - 2 > 0 and float_mask[ind[0], ind[1] - 2] > 0:
                        sum_2pix_extra[idx] = idx_mask_array[ind[0], ind[1] - 2]
                        sum_1pix_extra[idx] = idx_mask_array[ind[0], ind[1] - 1]
                    else:
                        test = idx_mask_array[ind[0], ind[1] - 1]

            # x+ direction
            if ind[0] + 1 < smask[0] - 1 and sum_1pix_extra[idx] == -1:
                if float_mask[ind[0] + 1, ind[1]] > 0:
                    if ind[0] + 2 < smask[0] - 1 and float_mask[ind[0] + 2, ind[1]] > 0:
                        sum_2pix_extra[idx] = idx_mask_array[ind[0] + 2, ind[1]]
                        sum_1pix_extra[idx] = idx_mask_array[ind[0] + 1, ind[1]]
                    else:
                        test = idx_mask_array[ind[0] + 1, ind[1]]
            
            # x- direction
            if ind[0] - 1 > 0 and sum_1pix_extra[idx] == -1:
                if float_mask[ind[0] - 1, ind[1]] > 0:
                    if ind[0] - 2 > 0 and float_mask[ind[0] - 2, ind[1]] > 0:
                        sum_2pix_extra[idx] = idx_mask_array[ind[0] - 2, ind[1]]
                        sum_1pix_extra[idx] = idx_mask_array[ind[0] - 1, ind[1]]
                    else:
                        test = idx_mask_array[ind[0] - 1, ind[1]]

            if sum_1pix_extra[idx] == -1 and test >= 0:
                sum_1pix_extra[idx] = test

    # Repeat for 2-pixel extra if specified
    if do_ext_2_pix:
        for idx in zip(*idx_2pix_extra):
            if np.sum(idx) * np.size(idx) != -1:
                ind = np.array(idx)
                test = -1

                # Similar extrapolation logic for 2-pixel extra

                # y+ direction
                if ind[1] + 2 < smask[1] - 1 and sum_2pix_extra[idx] == -1:
                    if float_mask[ind[0], ind[1] + 2] > 0:
                        if ind[1] + 3 < smask[1] - 1 and float_mask[ind[0], ind[1] + 3] > 0:
                            sum_2pix_extra[idx] = idx_mask_array[ind[0], ind[1] + 3]
                            sum_1pix_extra[idx] = idx_mask_array[ind[0], ind[1] + 2]
                        else:
                            test = idx_mask_array[ind[0], ind[1] + 2]

                # y+ direction
                if ind[1] - 2 > 0 and sum_2pix_extra[idx] == -1:
                    if float_mask[ind[0], ind[1] - 2] > 0:
                        if ind[1] - 3 > 0 and float_mask[ind[0], ind[1] - 3] > 0:
                            sum_2pix_extra[idx] = idx_mask_array[ind[0], ind[1] - 3]
                            sum_1pix_extra[idx] = idx_mask_array[ind[0], ind[1] - 2]
                        else:
                            test = idx_mask_array[ind[0], ind[1] - 2]

                # x+ direction
                if ind[0] + 2 < smask[0] - 1 and sum_2pix_extra[idx] == -1:
                    if float_mask[ind[0] + 2, ind[1]] > 0:
                        if ind[0] + 3 < smask[0] - 1 and float_mask[ind[0] + 3, ind[1]] > 0:
                            sum_2pix_extra[idx] = idx_mask_array[ind[0] + 3, ind[1]]
                            sum_1pix_extra[idx] = idx_mask_array[ind[0] + 2, ind[1]]
                        else:
                            test = idx_mask_array[ind[0] + 2, ind[1]]

                # x- direction
                if ind[0] - 2 > 0 and sum_2pix_extra[idx] == -1:
                    if float_mask[ind[0] - 2, ind[1]] > 0:
                        if ind[0] - 3 > 0 and float_mask[ind[0] - 3, ind[1]] > 0:
                            sum_2pix_extra[idx] = idx_mask_array[ind[0] - 3, ind[1]]
                            sum_1pix_extra[idx] = idx_mask_array[ind[0] - 2, ind[1]]
                        else:
                            test = idx_mask_array[ind[0] - 2, ind[1]]

                if sum_1pix_extra[idx] == -1 and test >= 0:
                    sum_1pix_extra[idx] = test

    # Find indices where extrapolation is needed (for 1-pixel extrapolation)
    # Find indices for 2-pixel extrapolation where value is defined
    idx_1pix = np.where(sum_1pix_extra >= 0)
    idxExtraPol2 = np.where(sum_2pix_extra[idx_1pix] >= 0)

    return (sum_1pix_extra.astype(np.int32),
           sum_2pix_extra.astype(np.int32),
           idx_1pix,
           idxExtraPol2)
           


