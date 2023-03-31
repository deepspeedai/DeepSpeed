# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .utils import *

try:
    import xgboost as xgb
except ImportError:
    xgb = None


class XGBoostCostModel():

    def __init__(self, loss_type, num_threads=None, log_interval=25, upper_model=None):

        assert xgb is not None, "missing requirements, please install deepspeed w. 'autotuning_ml' extra."

        self.loss_type = loss_type

        if loss_type == "reg":
            self.xgb_params = {
                "max_depth": 3,
                "gamma": 0.0001,
                "min_child_weight": 1,
                "subsample": 1.0,
                "eta": 0.3,
                "lambda": 1.0,
                "alpha": 0,
                "objective": "reg:linear",
            }
        elif loss_type == "rank":
            self.xgb_params = {
                "max_depth": 3,
                "gamma": 0.0001,
                "min_child_weight": 1,
                "subsample": 1.0,
                "eta": 0.3,
                "lambda": 1.0,
                "alpha": 0,
                "objective": "rank:pairwise",
            }
        else:
            raise RuntimeError("Invalid loss type: " + loss_type)

        self.xgb_params["verbosity"] = 0
        if num_threads:
            self.xgb_params["nthread"] = num_threads

    def fit(self, xs, ys):
        x_train = np.array(xs, dtype=np.float32)
        y_train = np.array(ys, dtype=np.float32)
        y_max = np.max(y_train)
        y_train = y_train / max(y_max, 1e-9)

        index = np.random.permutation(len(x_train))
        dtrain = xgb.DMatrix(x_train[index], y_train[index])

        self.bst = xgb.train(self.xgb_params, dtrain)

    def predict(self, xs):

        features = xgb.DMatrix(xs)

        return self.bst.predict(features)
