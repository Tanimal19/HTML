import sys
import time
import numpy as np
from scipy.sparse import csr_matrix
from libsvm.svmutil import *
from multiprocessing import Pool
import multiprocessing


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
            if label != 3 and label != 7:
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


def problem10():
    print(">> problem 10 start")

    result_file = "result10.csv"
    with open(f"hw6/{result_file}", "a") as f:
        f.write("C,Q,#sv\n")

    C_VAL = [0.1, 1, 10]
    Q_VAL = [2, 3, 4]

    for C in C_VAL:
        for Q in Q_VAL:
            print(f"\n--- C={C}, Q={Q} ---")

            print(f"start training...")
            start_time = time.time()

            prob = svm_problem(LABELS, FEATURES, isKernel=True)
            # -t 1: polynomial kernel, -d: degree, -c: cost
            param = svm_parameter(f"-t 1 -d {Q} -c {C} -g 1 -r 1 -q")
            m = svm_train(prob, param)

            print("finished time %ss" % (time.time() - start_time))

            sv_count = m.l
            print(f"#sv={sv_count}")

            with open(f"hw6/{result_file}", "a") as f:
                f.write(f"{C},{Q},{sv_count}\n")


def problem11():
    print(">> problem 11 start")

    result_file = "result11.csv"
    with open(f"hw6/{result_file}", "a") as f:
        f.write("C,gamma,margin\n")

    C_VAL = [0.1, 1, 10]
    GAMMA_VAL = [0.1, 1, 10]

    for C in C_VAL:
        for gamma in GAMMA_VAL:
            print(f"\n--- C={C}, gamma={gamma} ---")

            # if m is None:
            print(f"start training...")
            start_time = time.time()

            prob = svm_problem(LABELS, FEATURES, isKernel=True)
            # -t 2: radial basis function kernel, -g: gamma, -c: cost
            param = svm_parameter(f"-t 2 -g {gamma} -c {C} -q")
            m = svm_train(prob, param)

            print("finished time %ss" % (time.time() - start_time))

            alpha = np.array(m.get_sv_coef()).flatten()
            sv_idx = m.get_sv_indices()

            # |w| = \sum_{i=1}^{n} \alpha_i y_i x_i
            # exclude no sv, since alpha is 0
            w = np.zeros(len(sv_idx) + 1)
            for i, si in enumerate(sv_idx):
                x_i = FEATURES[si - 1]
                y_i = LABELS[si - 1]

                w_i = alpha[i] * y_i * x_i
                w[i] = np.linalg.norm(w_i)
            print(f"w={w}")

            margin = 1 / np.linalg.norm(w)
            print(f"margin={margin}")

            with open(f"hw6/{result_file}", "a") as f:
                f.write(f"{C},{gamma},{margin}\n")


def problem12():
    print(">> problem 12 start")

    result_file = "result12.csv"

    round = 128
    cpus = multiprocessing.cpu_count()

    inputs = []
    for i in range(1, round + 1):
        idx = np.random.permutation(LABELS.shape[0])
        idx_val = idx[:200]
        idx_train = idx[200:]

        for gamma in [0.01, 0.1, 1, 10, 100]:
            inputs.append((i, idx_val, idx_train, gamma))

    p = Pool(cpus)
    results = p.starmap(train_model, inputs)
    p.close()
    
    with open(f"hw6/{result_file}", "w") as f:
        f.write("round,gamma,eval\n")

        for (round, gamma, eval) in results:
            f.write(f"{round},{gamma},{eval}\n")


def train_model(round, idx_train, idx_val, gamma):
    print(f"task {round} gamma={gamma} start")
    prob = svm_problem(LABELS[idx_train], FEATURES[idx_train], isKernel=True)
    # -t 2: radial basis function kernel, -g: gamma, -c: cost
    param = svm_parameter(f"-t 2 -g {gamma} -c 1 -q")
    m = svm_train(prob, param)

    _, p_acc, _ = svm_predict(LABELS[idx_val], FEATURES[idx_val], m)
    eval = 1 - p_acc[0] / 100

    return (round, gamma, eval)


def main():
    if len(sys.argv) < 2:
        print("Usage: python hw6.py <problem>")
        return

    problem = sys.argv[1]

    global LABELS, FEATURES
    LABELS, FEATURES = read_data("data/mnist.scale.txt")

    root_start_time = time.time()

    if problem == "10":
        problem10()
    elif problem == "11":
        problem11()
    elif problem == "12":
        problem12()
    else:
        print("invalid problem number")

    print("\ntotal time %ss" % (time.time() - root_start_time))


main()
