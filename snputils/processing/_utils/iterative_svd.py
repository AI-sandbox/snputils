# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from sklearn.decomposition import TruncatedSVD
from sklearn.utils import check_array
import numpy as np
import random
import matplotlib.pyplot as plt
import copy
import logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)

from .solver import Solver
from .common import masked_mse

F32PREC = np.finfo(np.float32).eps


class IterativeSVD(Solver):
    def __init__(
            self,
            start_rank=1,
            end_rank=10,
            rank=5,
            choose_best=True,
            convergence_threshold=0.00001,
            max_iters=200,
            num_cores=1,
            svd_algorithm="arpack",
            init_fill_method="mean",
            min_value=None,
            max_value=None,
            verbose=True):
        Solver.__init__(
            self,
            fill_method=init_fill_method,
            min_value=min_value,
            max_value=max_value)
        self.start_rank = start_rank
        self.end_rank = end_rank
        self.rank = rank
        self.choose_best = choose_best
        self.max_iters = max_iters
        self.num_cores = num_cores
        self.svd_algorithm = svd_algorithm
        self.convergence_threshold = convergence_threshold
        self.verbose = verbose

    def _converged(self, X_old, X_new, missing_mask):
        # check for convergence
        old_missing_values = X_old[missing_mask]
        new_missing_values = X_new[missing_mask]
        difference = old_missing_values - new_missing_values
        ssd = np.sum(difference ** 2)
        old_norm_squared = (old_missing_values ** 2).sum()
        # edge cases
        if old_norm_squared == 0 or \
                (old_norm_squared < F32PREC and ssd > F32PREC):
            return False
        else:
            return (ssd / old_norm_squared) < self.convergence_threshold

    def create_validation_mask(self, X, missing_mask):
        observed_mask = ~missing_mask
        idx = np.flatnonzero(observed_mask)
        sampling = random.sample(list(idx), k=int(len(idx)*0.01))
        validation_mask = np.zeros(observed_mask.shape[0] * observed_mask.shape[1], dtype=bool)
        validation_mask[sampling] = True
        validation_mask = validation_mask.reshape(observed_mask.shape)
        return validation_mask
    
    def solve(self, X, missing_mask):
        if self.choose_best:
            cv_errors = []
            X = check_array(X, force_all_finite=False)
            observed_mask = ~missing_mask
            validation_mask = self.create_validation_mask(X, missing_mask)
            net_missing_mask = missing_mask + validation_mask
            net_observed_mask = ~net_missing_mask
            X_filled_init = copy.deepcopy(X)
            X_filled_init[net_missing_mask] = np.nan
            X_filled_init = super().fill(X=X_filled_init, missing_mask=net_missing_mask, fill_method=self.fill_method, inplace=True)

            global cross_validation
            def cross_validation(curr_rank):
                X_filled = copy.deepcopy(X_filled_init)
                for i in range(self.max_iters):
                    tsvd = TruncatedSVD(curr_rank, algorithm=self.svd_algorithm)
                    X_reduced = tsvd.fit_transform(X_filled)
                    X_reconstructed = tsvd.inverse_transform(X_reduced)
                    X_reconstructed = self.clip(X_reconstructed)
                    mse = masked_mse(
                        X_true=X,
                        X_pred=X_reconstructed,
                        mask=net_observed_mask)
                    if self.verbose:
                        logging.info(
                            "[IterativeSVD] Rank %d, Iter %d: observed MSE=%0.6f" % (
                                curr_rank, i, mse))
                    converged = self._converged(
                        X_old=X_filled,
                        X_new=X_reconstructed,
                        missing_mask=net_missing_mask)
                    X_filled[net_missing_mask] = X_reconstructed[net_missing_mask]
                    if converged:
                        break
                cv_error = masked_mse(
                    X_true=X,
                    X_pred=X_reconstructed,
                    mask=validation_mask)
                if self.verbose:
                    logging.info(
                        "[IterativeSVD] Rank %d: Cross-validation MSE=%0.6f" % (
                            curr_rank, cv_error))
                return cv_error

            ranks = np.arange(self.start_rank, self.end_rank+1)
#             pool = mp.Pool(self.num_cores)
#             cv_errors = pool.map(cross_validation, ranks)
#             pool.close()
            cv_errors = []
            for rank in ranks:
                cv_errors.append(cross_validation(rank))
            plt.plot(ranks, cv_errors)
            plt.xlabel('rank')
            plt.ylabel('CV mse')
            plt.savefig('IterativeSVD_CVplot.png')
            X_filled = copy.deepcopy(X)
            curr_rank = ranks[np.argmin(cv_errors)]
            if self.verbose:
                logging.info(
                    "[IterativeSVD] Best Rank chosen by cross-validation: %d" % (
                        curr_rank))
            for i in range(self.max_iters):
                tsvd = TruncatedSVD(curr_rank, algorithm=self.svd_algorithm)
                X_reduced = tsvd.fit_transform(X_filled)
                X_reconstructed = tsvd.inverse_transform(X_reduced)
                X_reconstructed = self.clip(X_reconstructed)
                mse = masked_mse(
                    X_true=X,
                    X_pred=X_reconstructed,
                    mask=observed_mask)
                if self.verbose:
                    logging.info(
                        "[IterativeSVD] Rank %d, Iter %d: observed MSE=%0.6f" % (
                            curr_rank, i, mse))
                converged = self._converged(
                    X_old=X_filled,
                    X_new=X_reconstructed,
                    missing_mask=missing_mask)
                X_filled[missing_mask] = X_reconstructed[missing_mask]
                if converged:
                    break
            return X_filled
        
        else:
            X = check_array(X, force_all_finite=False)
            observed_mask = ~missing_mask
            X_filled = copy.deepcopy(X)
            curr_rank = self.rank
            for i in range(self.max_iters):
                tsvd = TruncatedSVD(curr_rank, algorithm=self.svd_algorithm)
                X_reduced = tsvd.fit_transform(X_filled)
                X_reconstructed = tsvd.inverse_transform(X_reduced)
                X_reconstructed = self.clip(X_reconstructed)
                mse = masked_mse(
                    X_true=X,
                    X_pred=X_reconstructed,
                    mask=observed_mask)
                if self.verbose:
                    logging.info(
                        "[IterativeSVD] Rank %d, Iter %d: observed MSE=%0.6f" % (
                            curr_rank, i, mse))
                converged = self._converged(
                    X_old=X_filled,
                    X_new=X_reconstructed,
                    missing_mask=missing_mask)
                X_filled[missing_mask] = X_reconstructed[missing_mask]
                if converged:
                    break
            return X_filled
