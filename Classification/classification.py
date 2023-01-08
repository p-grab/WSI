import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random
from collections import Counter
import copy
from matplotlib import pyplot as plt

col_names = [
    "parents",
    "has_nurs",
    "form",
    "children",
    "housing",
    "finance",
    "social",
    "health",
    "class",
]


class Node:
    def __init__(
        self,
        feature=None,
        threshold=None,
        left_child=None,
        right_child=None,
        info_gain=None,
        value=None,
    ):

        self.feature = feature
        self.threshold = threshold
        self.left = left_child
        self.right = right_child
        self.info_gain = info_gain

        self.value = value


class DecisionTree:
    def __init__(self, max_depth=2):
        self.root = None

        self.max_depth = max_depth

    def add_node(self, data, curr_depth=0):
        sample_nr, feature_nr = np.shape(data[:, :-1])

        if curr_depth <= self.max_depth:
            best_split_params = self.get_best_split(data, sample_nr, feature_nr)
            if best_split_params["info_gain"] > 0:
                left_child = self.add_node(
                    best_split_params["left_data"], curr_depth + 1
                )
                right_child = self.add_node(
                    best_split_params["right_data"], curr_depth + 1
                )
                return Node(
                    best_split_params["feature"],
                    best_split_params["threshold"],
                    left_child,
                    right_child,
                    best_split_params["info_gain"],
                )

        Y = data[:, -1]
        Y = list(Y)
        leaf_value = max(Y, key=Y.count)
        return Node(value=leaf_value)

    def get_best_split(self, data, sample_nr, feature_nr):
        best_split_parametrs = {}
        best_split_parametrs["info_gain"] = 0

        max_info_gain = -float("inf")
        for feat in range(feature_nr):
            feats = data[:, feat]
            for threshold in np.unique(feats):
                left_data, right_data = self.split(data, feat, threshold)
                if len(left_data) > 0 and len(right_data) > 0:
                    y, left_y, right_y = (
                        data[:, -1],
                        left_data[:, -1],
                        right_data[:, -1],
                    )
                    info_gain = self.count_info_gain(
                        y,
                        left_y,
                        right_y,
                    )
                    if info_gain > max_info_gain:
                        best_split_parametrs["feature"] = feat
                        best_split_parametrs["threshold"] = threshold
                        best_split_parametrs["left_data"] = left_data
                        best_split_parametrs["right_data"] = right_data
                        best_split_parametrs["info_gain"] = info_gain

                        max_info_gain = info_gain
        return best_split_parametrs

    def split(self, data, feat, threshold):
        left = np.array([row for row in data if row[feat] == threshold])
        right = np.array([row for row in data if row[feat] != threshold])
        return left, right

    def count_info_gain(self, parent, l_child, r_child):

        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        gain = self.gini_index(parent) - (
            weight_l * self.gini_index(l_child) + weight_r * self.gini_index(r_child)
        )
        return gain

    def gini_index(self, y):
        classes = np.unique(y)
        gini = 0
        for cls in classes:
            p_cls = len(y[y == cls]) / len(y)
            gini += p_cls**2
        return 1 - gini

    def predict(self, X):
        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions

    def make_prediction(self, x, tree):
        if tree.value != None:
            return tree.value
        feature_val = x[tree.feature]
        if feature_val == tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)

    def print_tree(self, tree=None, indent=" "):
        if not tree:
            tree = self.root
        if tree.value is not None:
            print(tree.value)
        else:
            print(
                "X_" + str(col_names[tree.feature]),
                "=",
                tree.threshold,
                round(tree.info_gain, 2),
            )
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)


class DecisionForest:
    def __init__(self, trees=None) -> None:
        if trees is not None:
            self.trees = trees
        else:
            self.trees = []

    def make_forest(
        self, data, max_depth=4, trees_nr=15, elements_in_row=0, test_size=0.2
    ):
        X = data.iloc[:, :-1].values
        Y = data.iloc[:, -1].values.reshape(-1, 1)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=test_size, random_state=23
        )
        data_for_forest = np.concatenate((X_train, Y_train), axis=1)
        for i in range(trees_nr):
            data_to_choose = np.copy(data_for_forest)
            data_train_f = self.choose_data(data_to_choose, elements_in_row)

            tree = DecisionTree(max_depth=max_depth)
            tree.root = tree.add_node((data_train_f))

            self.trees.append(tree)

        return X_test, Y_test

    def choose_data(self, data_for_forest, elements_in_row):
        data_train = copy.deepcopy(data_for_forest)
        rows = [
            data_train[random.randint(0, len(data_train) - 1)]
            for _ in range(data_train.shape[0] - elements_in_row)
        ]
        data_train = np.array(rows)

        # a = [0, 1, 2, 3, 4, 5, 6, 7]
        # cols = random.sample(a, 3)
        # cols.append(-1)
        # print(cols)
        # data_train = np.array(data_train[:, cols])
        return data_train

    def predict_forest(self, X_test, Y_test):
        prediction = []
        for tree in self.trees:
            Y_pred = tree.predict(X_test)
            prediction.append(Y_pred)
        Y_pred_forest = self.most_common_at_position(prediction)
        return Y_pred_forest

    def most_common_at_position(self, arrays):
        result = []
        for i in range(len(arrays[0])):
            position_counts = Counter()
            for array in arrays:
                position_counts[array[i]] += 1
            result.append(position_counts.most_common(1)[0][0])
        return result


data = pd.read_csv("Classification/nursery.csv", header=None, names=col_names)
decisionforest = DecisionForest()
# elements_in_row if 0 -> 12960
X_test, Y_test = decisionforest.make_forest(
    data, max_depth=10, trees_nr=7, elements_in_row=int(12960 / 2), test_size=0.2
)
Y_pred_forest = decisionforest.predict_forest(X_test, Y_test)

from sklearn.metrics import accuracy_score

print(round(accuracy_score(Y_test, Y_pred_forest), 3))

occurences = {}
for x in set(Y_pred_forest):
    occurences[x] = [Y_pred_forest.count(x)]
print("Predict:")
print(occurences)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, Y_pred_forest)
print(cm)

from sklearn.metrics import classification_report

print(classification_report(Y_test, Y_pred_forest))

# courses = list(occurences.keys())
# val = [x for x in occurences.values()]
# values = []
# for i in range(len(val)):
#     values.append(val[i][0])
# print(values)
# print(courses)
# fig = plt.figure(figsize=(10, 7))
# plt.bar(courses, val)
