import sys
import numpy as np
from scipy.optimize import minimize

import utility_functions as uf
import matrices as mat


float_min = sys.float_info.min


def init_params(lmbda_max):
    """ 
    Initialize the parameters p and lmbda.
    p_{ii} is the probability of staying in the same class.
    p_{i,i+1} = 1 - p_{ii} is the probability of transitioning to the next class.
    lmbda_{ij} is a parameter for the entry in row i and column j of the transition matrix A.
    p has the structure [[p_11, p_12], [p_22, p_23], [p_33, p_34], [p_44, p_45]].
    lmbda has the structure [lmbda_12, lmbda_23, lmbda_34, lmbda_45].
    """
    lmbda = np.random.uniform(1, lmbda_max, 4)
    p = np.reshape(np.random.uniform(0, 1, 8), (4,2))
    return p, lmbda


def forward_backward(p, lmbda, lmbda_fix, ptime_diff, observed_seq, n_states=5):
    """
    Perform the forward-backward algorithm for current parameters p and lmbda.
    Compute P(Z_t = k |observed sequence).
    """
    A = mat.transition_matrix_fixed_lmbda(p, lmbda, lmbda_fix, ptime_diff)
    B = mat.expert_emission_matrix()
    pi = mat.start_prob()
    T = len(observed_seq)

    # Use normalized forward-backward algorithm to prevent underflow.
    
    # Calculate alpha.
    alphas = []
    norm = []
    
    alpha_1 = pi * B[:, observed_seq[0]]
    alphas.append(alpha_1 / np.sum(alpha_1))
    norm.append(np.sum(alpha_1))
    
    for t in range(1, T):
        alpha_t = B[:, observed_seq[t]] * (A[t-1].T @ alphas[t-1])
        alphas.append(alpha_t / np.sum(alpha_t))
        norm.append(np.sum(alpha_t))
    alpha = np.array(alphas)
    norm = np.array(norm)

    # Calculate beta.
    betas = []
    beta_T = np.ones(n_states)
    betas.append(beta_T)
    for t in range(1, T):
        beta_t = A[T-1-t] @ (B[:, observed_seq[T-t]] * betas[0])
        betas.insert(0, beta_t / norm[T-t])
    beta = np.array(betas)
    
    log_likelihood = np.sum(np.log(norm))

    return alpha, beta, log_likelihood


def E_step(p, lmbda, lmbda_fix, ptime_diff, observed_seq):
    """
    Given the observed sequence, compute E[I(Z_t=i, Z_{t+1}=j)], the expected number of 
    transitions i->j in the hidden sequence (Z_1, ..., Z_T). 
    """
    alpha, beta, loglik = forward_backward(p, lmbda, lmbda_fix, ptime_diff, observed_seq)
    A = mat.transition_matrix_fixed_lmbda(p, lmbda, lmbda_fix, ptime_diff)
    B = mat.expert_emission_matrix()
    T = len(observed_seq)

    xi = []
    for t in range(T-1):
        xi_t = np.diag(alpha[t]) @ A[t] @ np.diag(beta[t+1] * B[:, observed_seq[t+1]])
        xi_t = xi_t / (np.sum(xi_t))
        xi.append(xi_t)

    xi = np.array(xi)
    EN = np.sum(xi, axis = 0)

    return xi, EN


def log_likelihood(p, lmbda, lmbda_fix, ptime_diff, observed_seq):
    """
    Calculate the log-likelihood of the observed sequence.
    """
    _, _, loglik = forward_backward(p, lmbda, lmbda_fix, ptime_diff, observed_seq)
    return loglik


# Bounds and constraints for the optimization.
bnds = ((0, 1), (0, 1), (1, None))
cons = ({'type': 'eq', 'fun': lambda x: x[0] + x[1] - 1})


def M_step(p, lmbda, lmbda_fix, ptime_diff, observed_seq):
    """ 
    Do one maximization step of the EM algorithm.
    """
    xi, EN = E_step(p, lmbda, lmbda_fix, ptime_diff, observed_seq)
    
    def objective1(x):
        """
        Objective function to maximize for i=1.
        x = [p_11, p_12, lmbda_12]
        """
        nom_1 = x[0] * lmbda_fix[0] * np.exp(-lmbda_fix[0] * ptime_diff)
        nom_2 = x[1] * x[2] * np.exp(-x[2] * ptime_diff)
        denom = nom_1 + nom_2
        objective = np.sum(xi[:,0,0] * (np.log(nom_1 + float_min) - np.log(denom + float_min)) 
                           + xi[:,0,1] * (np.log(nom_2 + float_min) - np.log(denom + float_min)))
        return -objective  # - for minimization

    def objective2(x):
        """
        Objective function to maximize for i=2.
        x = [p_22, p_23, lmbda_23]
        """
        nom_1 = x[0] * lmbda_fix[1] * np.exp(-lmbda_fix[1] * ptime_diff)
        nom_2 = x[1] * x[2] * np.exp(-x[2] * ptime_diff)
        denom = nom_1 + nom_2
        objective = np.sum(xi[:,1,1] * (np.log(nom_1 + float_min) - np.log(denom + float_min)) 
                           + xi[:,1,2] * (np.log(nom_2 + float_min) - np.log(denom + float_min)))
        return -objective  # - for minimization

    def objective3(x):
        """
        Objective function to maximize for i=3.
        x = [p_33, p_34, lmbda_34]
        """
        nom_1 = x[0] * lmbda_fix[2] * np.exp(-lmbda_fix[2] * ptime_diff)
        nom_2 = x[1] * x[2] * np.exp(-x[2] * ptime_diff)
        denom = nom_1 + nom_2
        objective = np.sum(xi[:,2,2] * (np.log(nom_1 + float_min) - np.log(denom + float_min)) 
                           + xi[:,2,3] * (np.log(nom_2 + float_min) - np.log(denom + float_min)))
        return -objective  # - for minimization

    def objective4(x):
        """
        Objective function to maximize for i=4.
        x = [p_44, p_45, lmbda_45]
        """
        nom_1 = x[0] * lmbda_fix[3] * np.exp(-lmbda_fix[3] * ptime_diff)
        nom_2 = x[1] * x[2] * np.exp(-x[2] * ptime_diff)
        denom = nom_1 + nom_2
        objective = np.sum(xi[:,3,3] * (np.log(nom_1 + float_min) - np.log(denom + float_min)) 
                           + xi[:,3,4] * (np.log(nom_2 + float_min) - np.log(denom + float_min)))
        return -objective  # - for minimization
    
    x1 = np.append(p[0], lmbda[0])
    x2 = np.append(p[1], lmbda[1])
    x3 = np.append(p[2], lmbda[2])
    x4 = np.append(p[3], lmbda[3])
    
    method = 'SLSQP'
    res1 = minimize(objective1, x1, method=method,
                    bounds=bnds, constraints=cons, options={'ftol': 1e-10})
    res2 = minimize(objective2, x2, method=method,
                    bounds=bnds, constraints=cons, options={'ftol': 1e-10})
    res3 = minimize(objective3, x3, method=method,
                    bounds=bnds, constraints=cons, options={'ftol': 1e-10})
    res4 = minimize(objective4, x4, method=method,
                    bounds=bnds, constraints=cons, options={'ftol': 1e-10})
    
    p[0], p[1], p[2], p[3] = res1.x[:2], res2.x[:2], res3.x[:2], res4.x[:2]
    lmbda[0], lmbda[1], lmbda[2], lmbda[3] = res1.x[2], res2.x[2], res3.x[2], res4.x[2]
    
    obj_value = -(res1.fun + res2.fun + res3.fun + res4.fun)
    return p, lmbda, obj_value


def EM_algorithm(lmbda_fix, lmbda_max, ptime_diff, observed_seq, tol=1e-8):
        """
        Run update procedures of the EM-algorithm.
        """
        p_hist = []
        lmbda_hist = []
        loglik_hist = []
        
        p, lmbda = init_params(lmbda_max)
        p_hist.append(p.copy())
        lmbda_hist.append(lmbda.copy())
        loglik_hist.append(log_likelihood(p, lmbda, lmbda_fix, ptime_diff, observed_seq))
        
        counter = 1
        print('Iteration ' + str(counter) + '  ', end='\r')
        p, lmbda, obj_value = M_step(p, lmbda, lmbda_fix, ptime_diff, observed_seq)
        p_hist.append(p.copy())
        lmbda_hist.append(lmbda.copy())
        loglik_hist.append(log_likelihood(p, lmbda, lmbda_fix, ptime_diff, observed_seq))
        
        while (loglik_hist[-1] - loglik_hist[-2]) > tol and counter < 500:
            counter += 1
            print('Iteration ' + str(counter) + '  ', end='\r')
            p, lmbda, obj_value = M_step(p, lmbda, lmbda_fix, ptime_diff, observed_seq)
            p_hist.append(p.copy())
            lmbda_hist.append(lmbda.copy())
            loglik_hist.append(log_likelihood(p, lmbda, lmbda_fix, ptime_diff, observed_seq))
        return p, lmbda, p_hist, lmbda_hist, loglik_hist