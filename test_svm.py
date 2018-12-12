from svm import *
from helpers import *

# data = generate_training_data_binary(num=4)
# # plot_training_data_binary(data)
# w1, b1, S1 = svm_train_brute(data)
# plot_hyper_binary(w1, b1, data)

data = generate_training_data_multi(num=2)
svm = svm_train_multiclass(data)
plot_multi(data[0], svm[0], svm[1])

# dataOnly = data[0]
#
# for d in dataOnly:
#     y = svm_test_multiclass(svm[0], svm[1], d)
#     print("data point: ", d, "\tlabe: ", y)
