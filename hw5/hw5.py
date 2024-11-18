import sys
import numpy as np
from scipy.sparse import csr_matrix
from liblinear.liblinearutil import *


def read_data(filename):
    with open(filename) as f:
        lines = f.readlines()

        labels = []

        rows = []
        cols = []
        vals = []

        max_row = 0
        max_col = 0

        for line in lines:
            label = int(line.split()[0])
            if label != 2 and label != 6:
                continue
            labels.append(label)

            feature_row = line.split()[1:]
            for feature in feature_row:
                col, val = feature.split(":")

                rows.append(max_row)
                cols.append(int(col))
                vals.append(float(val))

                if int(col) > max_col:
                    max_col = int(col)

            max_row += 1

        # add bias term
        # for i in range(max_row):
        #     rows.append(i)
        #     cols.append(max_col)
        #     vals.append(1)
        # max_col += 1

        # change to numpy arrays
        labels = np.array(labels)
        features = csr_matrix(
            (vals, (rows, cols)), shape=(max_row, max_col + 1)
        ).toarray()

        print(f">> read {filename}")
        print("read labels: ", labels.shape)
        print("read features: ", features.shape)

    return labels, features


def problem10(result_file="result_10.csv", round=1):
    print("problem 10 start")

    labels, features = read_data("hw5/mnist.scale.txt")
    lables_t, features_t = read_data("hw5/mnist.scale.t.txt")

    lambdas = [0.01, 0.1, 1, 10, 100, 1000]

    for i in range(round):
        print(f"round {i}")

        # find best lambda with min ein
        min_ein = 1
        best_lambda = 0
        best_model = None

        for l in lambdas:
            # train
            cost = 1 / l
            prob = problem(labels, features)
            param = parameter(f"-s 6 -c {cost} -q")
            m = train(prob, param)

            _, p_acc, _ = predict(labels, features, m)
            ein = 1 - p_acc[0] / 100

            if ein < min_ein or (ein == min_ein and l > best_lambda):
                min_ein = ein
                best_lambda = l
                best_model = m

        print("best model: ", best_model.get_decfun()[0])

        # test on best lambda
        _, p_acc_t, _ = predict(lables_t, features_t, best_model)
        eout = 1 - p_acc_t[0] / 100

        # calculate non-zero components of model
        nz = np.count_nonzero(best_model.get_decfun()[0])

        with open(f"hw5/{result_file}", "a") as f:
            # lambda, eout, non-zero
            f.write(f"{best_lambda},{eout},{nz}\n")

    print("problem 10 done")


def problem11(result_file="result_11.csv", round=1):
    print("problem 11 start")

    labels, features = read_data("hw5/mnist.scale.txt")
    lables_t, features_t = read_data("hw5/mnist.scale.t.txt")

    lambdas = [0.01, 0.1, 1, 10, 100, 1000]

    for i in range(round):
        print(f"round {i}")

        # randomly select 8000 samples for training, others for validation
        idx = np.random.permutation(labels.shape[0])
        idx_train = idx[:8000]
        idx_val = idx[8000:]

        labels_train = labels[idx_train]
        features_train = features[idx_train]

        labels_val = labels[idx_val]
        features_val = features[idx_val]

        min_eval = 1
        best_lambda = 0

        for l in lambdas:
            # train
            cost = 1 / l
            prob = problem(labels_train, features_train)
            param = parameter(f"-s 6 -c {cost} -q")
            m = train(prob, param)

            _, p_acc, _ = predict(labels_val, features_val, m)
            eval_ = 1 - p_acc[0] / 100

            if eval_ < min_eval or (eval_ == min_eval and l > best_lambda):
                min_eval = eval_
                best_lambda = l

        # re-run training with best lambda on whole training set
        prob = problem(labels, features)
        param = parameter(f"-s 6 -c {1/best_lambda} -q")
        best_model = train(prob, param)

        _, p_acc_t, _ = predict(lables_t, features_t, best_model)
        eout = 1 - p_acc_t[0] / 100

        with open(f"hw5/{result_file}", "a") as f:
            # lambda, eout
            f.write(f"{best_lambda},{eout}\n")

    print("problem 11 done")


def problem12(result_file="result_12.csv", round=1):
    print("problem 12 start")

    labels, features = read_data("hw5/mnist.scale.txt")
    lables_t, features_t = read_data("hw5/mnist.scale.t.txt")

    lambdas = [0.01, 0.1, 1, 10, 100, 1000]

    for i in range(round):
        print(f"round {i}")

        # spilt data set 3-fold cross validation
        np.random.seed(i)
        idx = np.random.permutation(labels.shape[0])

        min_ecv = 1
        best_lambda = 0

        for l in lambdas:

            ecv = 1
            # 3-fold cross validation
            for m in range(3):
                idx_train = idx[
                    int(labels.shape[0] * m / 3) : int(labels.shape[0] * (m + 1) / 3)
                ]

                labels_train = labels[idx_train]
                features_train = features[idx_train]

                labels_val = labels[~idx_train]
                features_val = features[~idx_train]

                # train
                cost = 1 / l
                prob = problem(labels_train, features_train)
                param = parameter(f"-s 6 -c {cost} -q")
                m = train(prob, param)

                _, p_acc, _ = predict(labels_val, features_val, m)
                em = 1 - p_acc[0] / 100

                if em < ecv:
                    ecv = em

            if ecv < min_ecv or (ecv == min_ecv and l > best_lambda):
                min_ecv = ecv
                best_lambda = l

        # re-run training with best lambda on whole training set
        prob = problem(labels, features)
        param = parameter(f"-s 6 -c {1/best_lambda} -q")
        best_model = train(prob, param)

        _, p_acc_t, _ = predict(lables_t, features_t, best_model)
        eout = 1 - p_acc_t[0] / 100

        with open(f"hw5/{result_file}", "a") as f:
            # lambda, eout
            f.write(f"{best_lambda},{eout}\n")

    print("problem 12 done")


def main():
    if len(sys.argv) < 4:
        print("Usage: python hw5.py <problem> <round> <result_file>")
        return

    problem = sys.argv[1]
    round = int(sys.argv[2])
    result_file = sys.argv[3]

    if problem == "10":
        problem10(result_file, round)
    elif problem == "11":
        problem11(result_file, round)
    elif problem == "12":
        problem12(result_file, round)
    else:
        print("invalid problem number")


main()
