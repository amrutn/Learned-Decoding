
# Teacher HMM (fully known)
import numpy as np
import math
from typing import List

"""
K: hidden states
N: total states
pi: initial probability distribution over states
A, E: transition and emission matrices
"""

class HMMTeacher:
    def __init__(self, K=4, N=10):
        self.K, self.N = K, N
        self.pi = self._rand_simplex(K)
        self.A  = self._rand_stoch((K, K))
        self.E  = self._rand_stoch((K, N))

    def _rand_simplex(self, d):
        w = np.random.rand(d);  return w / w.sum()

    def _rand_stoch(self, shape):
        m = np.random.rand(*shape);  return m / m.sum(1, keepdims=True)

    # forward pass/filter
    def fwd(self, history: List[int]):
        """
        Return alpha (belief) after observing history.
        alpha is defined as: alpha_t (j) = P(s_t = j | x_{-T:0}), i.e posterior probability of 
        hidden state j given observed history.
        The forward pass has the formula:
        P(x_{t+1} = v | x_{-T:0}) = alpha_t A E
        """
        alpha = self.pi * self.E[:, history[0]]  
        alpha /= alpha.sum()
        for j in history[1:]:
            alpha = alpha @ self.A
            alpha = alpha * self.E[:, j] 
            alpha /= alpha.sum()
        return alpha

    def predictive(self, alpha):
        probs = (alpha @ self.A) @ self.E     
        return probs / probs.sum()

    # reward = log(p)
    def rollout_logp(self, context, future):
        alpha = self.fwd(context)
        log_p = 0.0
        for j in future:
            preds = self.predictive(alpha)
            log_p   += math.log(preds[j] + 1e-12)
            alpha = (alpha @ self.A); alpha *= self.E[:, j]; alpha /= alpha.sum()
        return log_p