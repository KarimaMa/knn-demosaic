from dciknn import DCI
import math
import numpy as np
import os
import argparse

def gen_patches(bayer, target,  patch_w, w, stride):
    c = 4 # bayer quad
    _ = math.ceil((args.w-args.patch_w) / args.stride)
    num_patches = _ * _
    patches = np.zeros((num_patches, c * patch_w * patch_w), dtype=np.float32)
    targets = np.zeros((num_patches, 8), dtype=np.float32)

    index = 0
    
    for i in range(0, w-patch_w, stride):
        for j in range(0, w-patch_w, stride):
            patches[index] = bayer[:,i:i+patch_w, j:j+patch_w].flatten()
            targets[index] = target[:,i+(patch_w//2), j+(patch_w//2)].flatten()
            index += 1

    return patches, targets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_ids", type=str)
    parser.add_argument("--test_ids", type=str)
    parser.add_argument("--patch_w", type=int)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--train_n", type=int, help="number of images in training set")
    parser.add_argument("--test_n", type=int, help="number of images in test set")
    parser.add_argument("--w", type=int, help="image size")
    args = parser.parse_args()

    # map from patch indices in dataset to file IDs
    c = 4 
    dim = args.patch_w * args.patch_w * c

    _ = math.ceil((args.w-args.patch_w) / args.stride)
    train_n = args.train_n * _ * _
    print(train_n)

    trainIndex2FileID = {}
    X = np.zeros((train_n, dim))
    Y = np.zeros((train_n, 8))
    cursor = 0
    for i, l in enumerate(open(args.train_ids, "r")):
        ID = l.strip()
        bayer_f = os.path.join(ID, "dense_bayer.data")
        bayer = np.fromfile(bayer_f, dtype=np.uint8).reshape((4, args.w, args.w)).astype(np.float32) / 255
        target_f = os.path.join(ID, "missing_bayer.data")
        target = np.fromfile(target_f, dtype=np.uint8).reshape((8, args.w, args.w)).astype(np.float32) / 255

        trainIndex2FileID[i] = ID

        patches, targets = gen_patches(bayer, target, args.patch_w, args.w, args.stride)
       
        num_patches = patches.shape[0]
        X[cursor:cursor+num_patches] = patches
        Y[cursor:cursor+num_patches] = targets
        cursor += num_patches

    assert(cursor == train_n)
    print("built data structure")

    num_comp_indices = 3
    num_simp_indices = 10
    num_levels = 4
    construction_field_of_view = 15
    construction_prop_to_retrieve = 0.004
    k = 4
    query_field_of_view = 120
    query_prop_to_retrieve = 0.85

    dci_db = DCI(dim, num_comp_indices, num_simp_indices)
    dci_db.add(X, num_levels = num_levels, field_of_view = construction_field_of_view, prop_to_retrieve = construction_prop_to_retrieve)

    # build query matrix
    test_n = args.test_n * _ * _
    testIndex2FileID = {}
    query = np.zeros((test_n, dim))
    queryY = np.zeros((test_n, 8))
    cursor = 0
    for i, l in enumerate(open(args.test_ids, "r")):
        ID = l.strip()
        testIndex2FileID[i] = ID
        bayer_f = os.path.join(ID, "dense_bayer.data")
        bayer = np.fromfile(bayer_f, dtype=np.uint8).reshape((4, args.w, args.w)).astype(np.float32) / 255
        target_f = os.path.join(ID, "missing_bayer.data")
        target = np.fromfile(target_f, dtype=np.uint8).reshape((8, args.w, args.w)).astype(np.float32) / 255

        patches, targets = gen_patches(bayer, target, args.patch_w, args.w, args.stride)
        num_patches = patches.shape[0]
        query[cursor:cursor+num_patches] = patches
        queryY[cursor:cursor+num_patches] = targets
        cursor += num_patches

    assert(cursor == test_n)     
    print("doing query")
    
    nn_idx, nn_dists = dci_db.query(query, num_neighbours = k, field_of_view = query_field_of_view, prop_to_retrieve = query_prop_to_retrieve)

    # evaluate accuracy
    dists_f = open("dists_at_stride{}_patch{}.txt".format(args.stride, args.patch_w), "a+")
    mse_f = open("mse_at_stride{}_patch{}.txt".format(args.stride, args.patch_w), "a+")
    losses = []
    for i in range(test_n):
        if i % 1000 == 0:
            print(i)

        avg_mse = 0
        for j in range(k):
            prediction_idx = nn_idx[i][j]
            prediction = Y[prediction_idx]
            actual = queryY[i]

            mse = ((prediction - actual)**2).mean()

            dists_f.write("{:.5f}\n".format(nn_dists[i][j]))
            mse_f.write("{:.5f}\n".format(mse))

            if j == 0:
                avg_mse = mse
            #avg_mse += (mse / k)

        losses += [avg_mse]

    total_mse = np.mean(losses)
    print("MSE: {:.4f}".format(total_mse))

