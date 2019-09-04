import sys
import numpy as np
from scipy.optimize import minimize

import utility_functions as uf
import matrices as mat


float_min = sys.float_info.min


def div(n, d):
    """
    Division by 0 equals to 0.
    """
    if len(np.shape(d)) > 0:
        res = np.zeros(len(d))
        for i in range(len(d)):
            if d[i]:
                res[i] = n[i] / d[i]
    else:
        if d:
            res = n / d
        else:
            res = 0
    return res


def hmt_init_params(lmbda_max):
    """ 
    Initialize the parameters p and lmbda.
    p_{ii} is the probability of staying in the same class.
    p_{ik} is the probability of transitioning from class i to class k.
    lmbda_{ij} is a parameter for the entry in row i and column j of the transition matrix A.
    """
    # 11 non-diagonal entries
    lmbda = np.random.uniform(1, lmbda_max, 11)
    # Number of entries for parameter p: 0:4, 1:3, 2:3, 3:2, 5:3, 8:2 (17 non-0 and non-1 entries)
    entry_n = [4, 3, 3, 2, 3, 2]
    p = []
    for n in entry_n:
        p.append(np.random.uniform(0, 1, n))
    return np.array(p), lmbda


def upward_downward(p, lmbda, lmbda_fix, X_obs, tree_obs, dev_map, n_states=10):
    """
    Compute P(Z_u = k |observed tree).
    """
    A = mat.hmt_transition_matrix(p, lmbda, lmbda_fix, X_obs, tree_obs)
    B = mat.hmt_expert_emission_matrix()
    pi = mat.hmt_start_prob()
    
    
    # Calculate hidden state marginal distributions.
    # Initialization for root node with pi
    marg = np.zeros((len(X_obs.index), n_states))
    root = tree_obs['root']
    marg[int(root)] = pi
    
    branches = ['branch S7-S5', 'branch S6-S5', 'branch S5-S3', 'branch S4-S3', 'branch S1-S3', 
                'branch S2-S1', 'branch S0-S1']
    for b in branches:
        branch = tree_obs[b]  # Indices from leaf to root
        if branch[-1] != root:
            parent = tree_obs['parents'][branch[-1]]
            marg[int(branch[-1])] = np.dot(marg[int(parent)], A[int(branch[-1])])
        for i in range(len(branch)-2, -1, -1):
            marg[int(branch[i])] = np.dot(marg[int(branch[i+1])], A[int(branch[i])])
    assert np.sum(marg, axis=1).all() == 1
            
            
    # Upward recursion
    # Initialization for leaves
    beta = np.zeros((len(X_obs.index), n_states))
    beta_uv = np.zeros((len(X_obs.index), n_states))
    norm = np.zeros(len(X_obs.index))
    for leaf in tree_obs['leaves']:
        beta[int(leaf)] = B[:, dev_map[X_obs.loc[leaf]['label']]] * marg[int(leaf)]
        norm[int(leaf)] = np.sum(beta[int(leaf)])
        beta[int(leaf)] /= norm[int(leaf)]

    branches = ['branch S6-S5', 'branch S4-S3', 'branch S2-S1', 'branch S0-S1']
    for b in branches:
        branch = tree_obs[b]
        for i in range(1, len(branch)):
            beta_uv_ = np.dot(div(beta[int(branch[i-1])], marg[int(branch[i-1])]), A[int(branch[i-1])].T)
            beta_uv[int(branch[i-1])] = beta_uv_
            beta[int(branch[i])] = (marg[int(branch[i])] * B[:, dev_map[X_obs.loc[branch[i]]['label']]] 
                                    * beta_uv_)
            norm[int(branch[i])] = np.sum(beta[int(branch[i])])
            beta[int(branch[i])] /= norm[int(branch[i])]
            
    # Branching point S1 '963'
    branching_point = '963'
    child = tree_obs['children'][branching_point]
    beta_uv1 = np.dot(div(beta[int(child[0])], marg[int(child[0])]), A[int(child[0])].T)
    beta_uv2 = np.dot(div(beta[int(child[1])], marg[int(child[1])]), A[int(child[1])].T)
    beta_uv[int(child[0])] = beta_uv1
    beta_uv[int(child[1])] = beta_uv2
    beta[int(branching_point)] = (marg[int(branching_point)] 
                                  * B[:, dev_map[X_obs.loc[branching_point]['label']]] * beta_uv1 * beta_uv2)
    norm[int(branching_point)] = np.sum(beta[int(branching_point)])
    beta[int(branching_point)] /= norm[int(branching_point)]
    
    # Branch S1-S3
    branch = tree_obs['branch S1-S3']
    for i in range(1, len(branch)):
        beta_uv_ = np.dot(div(beta[int(branch[i-1])], marg[int(branch[i-1])]), A[int(branch[i-1])].T)
        beta_uv[int(branch[i-1])] = beta_uv_
        beta[int(branch[i])] = (marg[int(branch[i])] * B[:, dev_map[X_obs.loc[branch[i]]['label']]] 
                                * beta_uv_)
        norm[int(branch[i])] = np.sum(beta[int(branch[i])])
        beta[int(branch[i])] /= norm[int(branch[i])]
        
    # Branching point S3 '509'
    branching_point = '509'
    child = tree_obs['children'][branching_point]
    beta_uv1 = np.dot(div(beta[int(child[0])], marg[int(child[0])]), A[int(child[0])].T)
    beta_uv2 = np.dot(div(beta[int(child[1])], marg[int(child[1])]), A[int(child[1])].T)
    beta_uv[int(child[0])] = beta_uv1
    beta_uv[int(child[1])] = beta_uv2
    beta[int(branching_point)] = (marg[int(branching_point)] 
                                  * B[:, dev_map[X_obs.loc[branching_point]['label']]] 
                                  * beta_uv1 * beta_uv2)
    norm[int(branching_point)] = np.sum(beta[int(branching_point)])
    beta[int(branching_point)] /= norm[int(branching_point)]
    
    # Branch S5-S3
    branch = tree_obs['branch S5-S3']
    for i in range(1, len(branch)):
        beta_uv_ = np.dot(div(beta[int(branch[i-1])], marg[int(branch[i-1])]), A[int(branch[i-1])].T)
        beta_uv[int(branch[i-1])] = beta_uv_
        beta[int(branch[i])] = (marg[int(branch[i])] * B[:, dev_map[X_obs.loc[branch[i]]['label']]] 
                                * beta_uv_)
        norm[int(branch[i])] = np.sum(beta[int(branch[i])])
        beta[int(branch[i])] /= norm[int(branch[i])]
        
    # Branching point S5 '1194'
    branching_point = '1194'
    child = tree_obs['children'][branching_point]
    beta_uv1 = np.dot(div(beta[int(child[0])], marg[int(child[0])]), A[int(child[0])].T)
    beta_uv2 = np.dot(div(beta[int(child[1])], marg[int(child[1])]), A[int(child[1])].T)
    beta_uv[int(child[0])] = beta_uv1
    beta_uv[int(child[1])] = beta_uv2
    beta[int(branching_point)] = (marg[int(branching_point)] 
                                  * B[:, dev_map[X_obs.loc[branching_point]['label']]] 
                                  * beta_uv1 * beta_uv2)
    norm[int(branching_point)] = np.sum(beta[int(branching_point)])
    beta[int(branching_point)] /= norm[int(branching_point)]
    
    # Branch S7-S5
    branch = tree_obs['branch S7-S5']
    for i in range(1, len(branch)):
        beta_uv_ = np.dot(div(beta[int(branch[i-1])], marg[int(branch[i-1])]), A[int(branch[i-1])].T)
        beta_uv[int(branch[i-1])] = beta_uv_
        beta[int(branch[i])] = (marg[int(branch[i])] * B[:, dev_map[X_obs.loc[branch[i]]['label']]]
                                * beta_uv_)
        norm[int(branch[i])] = np.sum(beta[int(branch[i])])
        beta[int(branch[i])] /= norm[int(branch[i])]
    
 
    # Calculate log-likelihood.
    loglik = np.sum(np.log(norm))
    
    
    # Downward recursion
    # Initialization for root node
    gamma = np.zeros((len(X_obs.index), n_states))
    gamma[int(root)] = beta[int(root)]
    
    branches = ['branch S7-S5', 'branch S6-S5', 'branch S5-S3', 'branch S4-S3', 'branch S1-S3', 
                'branch S2-S1', 'branch S0-S1']
    
    for b in branches:
        branch = tree_obs[b]
        if branch[-1] != root:
            parent = tree_obs['parents'][branch[-1]]
            gamma[int(branch[-1])] = (div(beta[int(branch[-1])], marg[int(branch[-1])])
                                      * np.dot(div(gamma[int(parent)], beta_uv[int(branch[-1])]), 
                                               A[int(branch[-1])]))
        for i in range(len(branch)-2, -1, -1):
            gamma[int(branch[i])] = (div(beta[int(branch[i])], marg[int(branch[i])])
                                     * np.dot(div(gamma[int(branch[i+1])], beta_uv[int(branch[i])]), 
                                              A[int(branch[i])]))
                    
    return marg, beta, beta_uv, gamma, loglik


def hmt_E_step(p, lmbda, lmbda_fix, X_obs, tree_obs, dev_map, n_states=10):
    """
    Given the observed tree, compute E[I(Z_t=i, Z_{t+1}=j)], the expected number of 
    transitions i->j in the hidden tree (Z_1, ..., Z_U). 
    """
    marg, beta, beta_uv, gamma, loglik = upward_downward(p, lmbda, lmbda_fix, X_obs, tree_obs, dev_map)
    A = mat.hmt_transition_matrix(p, lmbda, lmbda_fix, X_obs, tree_obs)
    
    xi = np.zeros((len(X_obs.index), n_states, n_states))
    branches = ['branch S7-S5', 'branch S6-S5', 'branch S5-S3', 'branch S4-S3', 'branch S1-S3', 
                'branch S2-S1', 'branch S0-S1']
    root = tree_obs['root']
    for b in branches:
        branch = tree_obs[b]
        if branch[-1] != root:
            parent = tree_obs['parents'][branch[-1]]
            for k in range(n_states):
                for l in range(n_states):
                    xi[int(branch[-1]), k, l] = (div(gamma[int(parent), k], beta_uv[int(branch[-1]), k])
                                                 * div(beta[int(branch[-1]), l], marg[int(branch[-1]), l]) 
                                                 * A[int(branch[-1])][k, l])
        for i in range(len(branch)-2, -1, -1):
            for k in range(n_states):
                for l in range(n_states):
                    xi[int(branch[i]), k, l] = (div(gamma[int(branch[i+1]), k], beta_uv[int(branch[i]), k])
                                                * div(beta[int(branch[i]), l], marg[int(branch[i]), l])
                                                * A[int(branch[i])][k, l])
    
    return xi


def hmt_log_likelihood(p, lmbda, lmbda_fix, X_obs, tree_obs, dev_map):
    """
    Calculate the log-likelihood of the observed tree.
    """
    _, _, _, _, loglik = upward_downward(p, lmbda, lmbda_fix, X_obs, tree_obs, dev_map)
    return loglik


def hmt_M_step(p, lmbda, lmbda_fix, X_obs, tree_obs, pt_diff, dev_map):
    """ 
    Do one maximization step of the EM algorithm.
    """
    xi = hmt_E_step(p, lmbda, lmbda_fix, X_obs, tree_obs, dev_map)
    float_min = sys.float_info.min
    root = int(tree_obs['root'])
    xi = np.delete(xi, root, 0)
    
    def objective0(x):
        """
        Objective function to maximize for i=0.
        x = [p_00, p_01, p_06, p_07, lmbda_01, lmbda_06, lmbda_07]
        """
        
        nom_1 = x[0] * lmbda_fix[0] * np.exp(-lmbda_fix[0] * pt_diff)
        nom_2 = x[1] * x[4] * np.exp(-x[4] * pt_diff)
        nom_3 = x[2] * x[5] * np.exp(-x[5] * pt_diff)
        nom_4 = x[3] * x[6] * np.exp(-x[6] * pt_diff)
        denom = nom_1 + nom_2 + nom_3 + nom_4
        objective = np.sum(xi[:,0,0] * (np.log(nom_1 + float_min) - np.log(denom + float_min)) 
                           + xi[:,0,1] * (np.log(nom_2 + float_min) - np.log(denom + float_min))
                           + xi[:,0,6] * (np.log(nom_3 + float_min) - np.log(denom + float_min))
                           + xi[:,0,7] * (np.log(nom_4 + float_min) - np.log(denom + float_min)))
        return -objective  # - for minimization

    def objective1(x):
        """
        Objective function to maximize for i=1.
        x = [p_11, p_12, p_17, lmbda_12, lmbda_17]
        """
        nom_1 = x[0] * lmbda_fix[1] * np.exp(-lmbda_fix[1] * pt_diff)
        nom_2 = x[1] * x[3] * np.exp(-x[3] * pt_diff)
        nom_3 = x[2] * x[4] * np.exp(-x[4] * pt_diff)
        denom = nom_1 + nom_2 + nom_3
        objective = np.sum(xi[:,1,1] * (np.log(nom_1 + float_min) - np.log(denom + float_min)) 
                           + xi[:,1,2] * (np.log(nom_2 + float_min) - np.log(denom + float_min))
                           + xi[:,1,7] * (np.log(nom_3 + float_min) - np.log(denom + float_min)))
        return -objective  # - for minimization
    
    def objective2(x):
        """
        Objective function to maximize for i=2.
        x = [p_22, p_23, p_27, lmbda_23, lmbda_27]
        """
        nom_1 = x[0] * lmbda_fix[2] * np.exp(-lmbda_fix[2] * pt_diff)
        nom_2 = x[1] * x[3] * np.exp(-x[3] * pt_diff)
        nom_3 = x[2] * x[4] * np.exp(-x[4] * pt_diff) 
        denom = nom_1 + nom_2 + nom_3
        objective = np.sum(xi[:,2,2] * (np.log(nom_1 + float_min) - np.log(denom + float_min)) 
                           + xi[:,2,3] * (np.log(nom_2 + float_min) - np.log(denom + float_min))
                           + xi[:,2,7] * (np.log(nom_3 + float_min) - np.log(denom + float_min)))
        return -objective  # - for minimization

    def objective3(x):
        """
        Objective function to maximize for i=3.
        x = [p_33, p_34, lmbda_34]
        """
        nom_1 = x[0] * lmbda_fix[3] * np.exp(-lmbda_fix[3] * pt_diff)
        nom_2 = x[1] * x[2] * np.exp(-x[2] * pt_diff)
        denom = nom_1 + nom_2
        objective = np.sum(xi[:,3,3] * (np.log(nom_1 + float_min) - np.log(denom + float_min)) 
                           + xi[:,3,4] * (np.log(nom_2 + float_min) - np.log(denom + float_min)))
        return -objective  # - for minimization
    
    def objective5(x):
        """
        Objective function to maximize for i=5.
        x = [p_55, p_50, p_58, lmbda_50, lmbda_58]
        """
        nom_1 = x[0] * lmbda_fix[5] * np.exp(-lmbda_fix[5] * pt_diff)
        nom_2 = x[1] * x[3] * np.exp(-x[3] * pt_diff)
        nom_3 = x[2] * x[4] * np.exp(-x[4] * pt_diff)
        denom = nom_1 + nom_2 + nom_3
        objective = np.sum(xi[:,5,5] * (np.log(nom_1 + float_min) - np.log(denom + float_min)) 
                           + xi[:,5,0] * (np.log(nom_2 + float_min) - np.log(denom + float_min))
                           + xi[:,5,8] * (np.log(nom_3 + float_min) - np.log(denom + float_min)))
        return -objective  # - for minimization
    
    def objective8(x):
        """
        Objective function to maximize for i=8.
        x = [p_88, p_89, lmbda_89]
        """
        nom_1 = x[0] * lmbda_fix[8] * np.exp(-lmbda_fix[8] * pt_diff)
        nom_2 = x[1] * x[2] * np.exp(-x[2] * pt_diff)
        denom = nom_1 + nom_2
        objective = np.sum(xi[:,8,8] * (np.log(nom_1 + float_min) - np.log(denom + float_min)) 
                           + xi[:,8,9] * (np.log(nom_2 + float_min) - np.log(denom + float_min)))
        return -objective  # - for minimization

    
    bnds0 = ((0, 1), (0, 1), (0, 1), (0, 1), (1, None), (1, None), (1, None))
    cons0 = ({'type': 'eq', 'fun': lambda x: x[0] + x[1] + x[2] + x[3] - 1})
    
    bnds1 = ((0, 1), (0, 1), (1, None))
    cons1 = ({'type': 'eq', 'fun': lambda x: x[0] + x[1] - 1})
    
    bnds2 = ((0, 1), (0, 1), (0, 1), (1, None), (1, None))
    cons2 = ({'type': 'eq', 'fun': lambda x: x[0] + x[1] + x[2] - 1})
    
    x0 = np.append(p[0], lmbda[:3])
    x1 = np.append(p[1], lmbda[3:5])
    x2 = np.append(p[2], lmbda[5:7])
    x3 = np.append(p[3], lmbda[7])
    x5 = np.append(p[4], lmbda[8:10])
    x8 = np.append(p[5], lmbda[10])
    
    method = 'SLSQP'
    res0 = minimize(objective0, x0, method=method,
                    bounds=bnds0, constraints=cons0, options={'ftol': 1e-10})
    res1 = minimize(objective1, x1, method=method,
                    bounds=bnds2, constraints=cons2, options={'ftol': 1e-10})
    res2 = minimize(objective2, x2, method=method,
                    bounds=bnds2, constraints=cons2, options={'ftol': 1e-10})
    res3 = minimize(objective3, x3, method=method,
                    bounds=bnds1, constraints=cons1, options={'ftol': 1e-10})
    res5 = minimize(objective5, x5, method=method,
                    bounds=bnds2, constraints=cons2, options={'ftol': 1e-10})
    res8 = minimize(objective8, x8, method=method,
                    bounds=bnds1, constraints=cons1, options={'ftol': 1e-10})
    
    p[0], p[1], p[2], p[3], p[4], p[5] = res0.x[:4], res1.x[:3], res2.x[:3], res3.x[:2], res5.x[:3], res8.x[:2]
    lmbda[:3], lmbda[3:5], lmbda[5:7], lmbda[7] = res0.x[4:], res1.x[3:], res2.x[3:], res3.x[2]
    lmbda[8:10], lmbda[10] = res5.x[3:], res8.x[2]

    return p, lmbda


def hmt_EM_algorithm(lmbda_fix, lmbda_max, X_obs, tree_obs, pt_diff, dev_map, tol=1e-8):
        """
        Run update procedures of the EM-algorithm.
        """
        p_hist = []
        lmbda_hist = []
        loglik_hist = []
        
        p, lmbda = hmt_init_params(lmbda_max)
        p_hist.append(p.copy())
        lmbda_hist.append(lmbda.copy())
        loglik_hist.append(hmt_log_likelihood(p, lmbda, lmbda_fix, X_obs, tree_obs, dev_map))
        
        counter = 1
        print('Iteration ' + str(counter) + '  ', end='\r')
        p, lmbda = hmt_M_step(p, lmbda, lmbda_fix, X_obs, tree_obs, pt_diff, dev_map)
        p_hist.append(p.copy())
        lmbda_hist.append(lmbda.copy())
        loglik_hist.append(hmt_log_likelihood(p, lmbda, lmbda_fix, X_obs, tree_obs, dev_map))
        
        while (loglik_hist[-1] - loglik_hist[-2]) > tol and counter < 500:
            counter += 1
            print('Iteration ' + str(counter) + '  ', end='\r')
            p, lmbda = hmt_M_step(p, lmbda, lmbda_fix, X_obs, tree_obs, pt_diff, dev_map)
            p_hist.append(p.copy())
            lmbda_hist.append(lmbda.copy())
            loglik_hist.append(hmt_log_likelihood(p, lmbda, lmbda_fix, X_obs, tree_obs, dev_map))
        return p, lmbda, p_hist, lmbda_hist, loglik_hist