import sys
import time
import numpy as np
from libsvm.svmutil import *
from multiprocessing import Pool
import multiprocessing


def decision_stump_task(feature, threshold, polarity, X, y, weights):
    predictions = polarity * np.sign(X[:, feature].toarray().flatten() - threshold)
    predictions[predictions == 0] = -1

    weighted_error = np.sum(weights[predictions != y])

    return (feature, threshold, polarity, weighted_error)


def decision_stump(X, y, weights):
    n_samples, n_features = X.shape
    best_feature, best_threshold, best_polarity = None, None, None
    min_weighted_error = float('inf')

    pool_inputs = []

    for feature in range(n_features):
        thresholds = np.unique(X[:, feature].toarray().flatten())
        for threshold in thresholds:
            for polarity in [1, -1]:
                pool_inputs.append((feature, threshold, polarity, X, y, weights))

    cpus = multiprocessing.cpu_count()
    p = Pool(int(cpus/4))
    results = p.starmap(decision_stump_task, pool_inputs)
    p.close()

    for (feature, threshold, polarity, weighted_error) in results:
        if weighted_error < min_weighted_error:
            best_feature = feature
            best_threshold = threshold
            best_polarity = polarity
            min_weighted_error = weighted_error

    return best_feature, best_threshold, best_polarity, min_weighted_error



def adaboost(X_train, y_train, X_test, y_test):
    n_train, n_features = X_train.shape
    n_test = X_test.shape[0]

    weights = np.ones(n_train) / n_train

    gt_list = []
    alpha_list = []


    result10 = {
        "ein_history": [],
        "epsilon_history": [],
    }
    result11 = {
        "ein_history": [],
        "eout_history": [],
    }
    result12 = {
        "ut_history": [],
        "ein_history": [],
    }

    for t in range(500):
        start_time = time.time()

        # get gt by decision stump
        feature, threshold, polarity, weighted_error = decision_stump(X_train, y_train, weights)
        gt_list.append((feature, threshold, polarity))

        # compute alpha (gt weight)
        alpha = 0.5 * np.log((1 - weighted_error) / max(weighted_error, 1e-10))
        alpha_list.append(alpha)

        x = X_train[:, feature].toarray().flatten()

        # update weights
        predictions = polarity * np.sign(x - threshold)
        predictions[predictions == 0] = -1
        weights *= np.exp(-alpha * y_train * predictions)
        weights /= np.sum(weights)

        # # Ein(gt)
        # ein_gt = np.mean(predictions != y_train)            
        # result10["ein_history"].append(ein_gt)

        # # epsilon
        # epsilon_t = weighted_error / np.sum(weights)
        # result10["epsilon_history"].append(epsilon_t)

        # Ein(Gt)
        predictions = np.zeros(n_train)
        for i in range(len(gt_list)):
            feature, threshold, polarity = gt_list[i]
            alpha = alpha_list[i]
            x = X_train[:, feature].toarray().flatten()
            predictions += alpha_list[i] * (polarity * np.sign(x - threshold))
        predictions = np.sign(predictions)
        predictions[predictions == 0] = -1
        ein_Gt = np.mean(predictions != y_train)
        # result11["ein_history"].append(ein_Gt)
        result12["ein_history"].append(ein_Gt)

        # # Eout(Gt)
        # predictions = np.zeros(n_test)
        # for i in range(len(gt_list)):
        #     feature, threshold, polarity = gt_list[i]
        #     alpha = alpha_list[i]
        #     x = X_test[:, feature].toarray().flatten()
        #     predictions += alpha_list[i] * (polarity * np.sign(x - threshold))
        # predictions = np.sign(predictions)
        # predictions[predictions == 0] = -1
        # eout_t = np.mean(predictions != y_test)
        # result11["eout_history"].append(eout_t)

        # For problem 12
        # Ut
        ut = np.sum(weights)
        result12["ut_history"].append(ut)

        print(f"[{t}]\tused {time.time() - start_time}s")
        print(f"\tein_gt={ein_gt},\tepsilon={epsilon_t},\tein_Gt={ein_Gt},\teout_Gt={eout_t},\tUt={ut}")

    return result10, result11, result12

root_start_time = time.time()

y_train, X_train = svm_read_problem("madelon.txt", return_scipy=True)
y_test, X_test = svm_read_problem("madelon.t.txt", return_scipy=True)

print(X_train.shape)
print(X_test.shape)

result10, result11, result12 = adaboost(X_train, y_train, X_test, y_test)

# with open("result10.csv", "w") as f:
#     f.write("Ein,epsilon\n")
#     for ein, epsilon in zip(result10["ein_history"], result10["epsilon_history"]):
#         f.write(f"{ein},{epsilon}\n")

# with open("result11.csv", "w") as f:
#     f.write("Ein,Eout\n")
#     for ein, eout in zip(result11["ein_history"], result11["eout_history"]):
#         f.write(f"{ein},{eout}\n")

with open("result12.csv", "w") as f:
    f.write("Ut,Ein\n")
    for ut, ein in zip(result12["ut_history"], result12["ein_history"]):
        f.write(f"{ut},{ein}\n")

print("Done in %.2fs" % (time.time() - root_start_time))