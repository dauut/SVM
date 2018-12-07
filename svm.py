import numpy as np
import matplotlib.pyplot as plt
import sys
import helpers as hp


def svm_train_brute(training_data):
    # convert data to np array just in case
    training_data = np.asarray(training_data)

    positive = training_data[training_data[:, 2] == 1]
    negative = training_data[training_data[:, 2] == -1]

    is_first = True
    margin = 0
    s, w, b = None, None, None

    # in 2D data we need only 2 or 3 support vectors
    # we will look every couple labels for finding the max margin

    # one positive - one negative
    for pos in positive:
        for neg in negative:
            mid_point = (pos[0:2] + neg[0:2]) / 2
            w = np.array(pos[:-1] - neg[:-1])
            w = w / np.sqrt((w[0] * w[0] + w[1] * w[1]))
            b = -1 * (w[0] * mid_point[0] + w[1] * mid_point[1])

            if is_first:
                margin = compute_margin(training_data, w, b)
                s = np.array([pos, neg])
                w = w
                b = b
                is_first = False

            elif margin <= compute_margin(training_data, w, b):
                margin = compute_margin(training_data, w, b)
                s = np.array([pos, neg])
                w = w
                b = b

    # for pos in positive:
    #     for neg in negative:
    #         for neg1 in negative:
    #             if (pos[0] != neg[0]) and (pos[1] != neg[1]):


    return w, b, s


def distance_point_to_hyperplane(pt, w, b):
    # print(np.abs(((pt[0] * w[0]) + (pt[1] * w[1]) + b) / (np.sqrt((w[0] * w[0]) + (w[1] * w[1])))))
    return np.abs(((pt[0] * w[0]) + (pt[1] * w[1]) + b) / (np.sqrt((w[0] * w[0]) + (w[1] * w[1]))))


def compute_margin(data, w, b):
    margin = distance_point_to_hyperplane(data[0, :-1], w, b)

    for pt in data:
        distance = distance_point_to_hyperplane(pt[:-1], w, b)
        if distance < margin:
            margin = distance_point_to_hyperplane(pt[:-1], w, b)
        if svm_test_brute(w, b, pt) != pt[2]:
            return 0

    return margin


def svm_test_brute(w, b, x):
    if np.dot(w, x[:-1]) + b > 0:
        return 1
    else:
        return -1


def svm_train_multiclass(training_data):
    print()


def svm_test_multiclass(W, B, x):
    print()


data = hp.generate_training_data_binary(num=1)
# data = hp.generate_training_data_binary(num=2)
# hp.plot_training_data_binary(data)
svm_train_brute(data)
