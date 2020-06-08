import numpy as np


def start_prob():
    """
    Initial probabilities for the hidden Markov model.
    Five classes: ['PMY', 'MY', 'MMY', 'BNE', 'SNE']
    """
    return [0.9, 0.025, 0.025, 0.025, 0.025]


def expert_emission_matrix():
    """
    Emission matrix based on estimations by an expert.
    Five classes: ['PMY', 'MY', 'MMY', 'BNE', 'SNE']
    """
    emission = np.array([[0.7, 0.25, 0.04, 0.005, 0.005],
                         [0.23, 0.52, 0.24, 0.005, 0.005],
                         [0.03, 0.17, 0.75, 0.045, 0.005],
                         [0.005, 0.005, 0.03, 0.82, 0.14],
                         [0.005, 0.005, 0.005, 0.065, 0.92]])
    assert (emission.sum(axis=1)==1).all()
    return emission


def transition_matrix_fixed_lmbda(p, lmbda, lmbda_fix, ptime_diff):
    """
    Calculate the transition matrices using the pseudotime difference between cells. 
    p and lmbda are parameters for the entries.
    lmbda_fix is fixed.
    """
    transition = []
    
    for i in range(len(ptime_diff)):
        A1 = np.diag(np.append(p[:,0] * lmbda_fix * np.exp(-lmbda_fix * ptime_diff[i]), 1))
        A2 = np.diag(p[:,1] * lmbda * np.exp(-lmbda * ptime_diff[i]), 1)
        A = (A1+A2) / (A1+A2).sum(axis=1)[:, None]
        transition.append(A)
    return transition


def hmt_start_prob():
    """
    Initial probabilities for the hidden Markov tree.
    Ten classes: ['PMY', 'MY', 'MMY', 'BNE', 'SNE', 'BL', 'BA', 'EO', 'LY', 'PC']
    """
    return np.array([0.005, 0.005, 0.005, 0.005, 0.005, 0.955, 0.005, 0.005, 0.005, 0.005])


def hmt_expert_emission_matrix():
    """
    Emission matrix based on estimations by an expert.
    Ten classes: ['PMY', 'MY', 'MMY', 'BNE', 'SNE', 'BL', 'BA', 'EO', 'LY', 'PC']
    """
    emission = np.array([[0.7, 0.23, 0.03, 0.005, 0.005, 0.01, 0.005, 0.005, 0.005, 0.005],
                         [0.22, 0.52, 0.225, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005],
                         [0.02, 0.16, 0.75, 0.04, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005],
                         [0.005, 0.005, 0.02, 0.82, 0.125, 0.005, 0.005, 0.005, 0.005, 0.005],
                         [0.005, 0.005, 0.005, 0.04, 0.92, 0.005, 0.005, 0.005, 0.005, 0.005],
                         [0.06, 0.005, 0.005, 0.005, 0.005, 0.8, 0.005, 0.005, 0.105, 0.005],
                         [0.005, 0.005, 0.005, 0.005, 0.02, 0.005, 0.94, 0.005, 0.005, 0.005],
                         [0.005, 0.005, 0.005, 0.005, 0.01, 0.005, 0.005, 0.95, 0.005, 0.005],
                         [0.005, 0.005, 0.005, 0.005, 0.005, 0.055, 0.005, 0.005, 0.9, 0.01],
                         [0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.06, 0.9]])
    assert (emission.sum(axis=1)==1).all()
    assert emission.shape == (10, 10)
    return emission


def hmt_transition_matrix(p, lmbda, lmbda_fix, X_obs, tree_obs):
    """
    Calculate the transition matrices using the pseudotime difference between cells. 
    p and lmbda are parameters for the entries.
    lmbda_fix is fixed.
    X_obs and tree_obs contain information about the Markov tree.
    """
    entry_idx = [[0,1], [0,6], [0,7], [1,2], [1,7], [2,3], [2,7], [3,4], [5,0], [5,8], [8,9]]
    p_transition = [p[0][1], p[0][2], p[0][3], p[1][1], p[1][2], p[2][1], p[2][2], p[3][1], p[4][1], p[4][2], p[5][1]]
    
    tm = []
    # Every cell has a transition matrix depending on the pseudotime difference between the pseudotime of 
    # the cell and the pseudotime of its parent.
    for i in range(len(X_obs.index)):
        parent = tree_obs['parents'][str(i)]
        if parent != None:
            ptime_diff = X_obs.loc[str(i)]['S7_pseudotime'] - X_obs.loc[parent]['S7_pseudotime']
            p_diag = np.array([p[0][0], p[1][0], p[2][0], p[3][0], 1, p[4][0], 1, 1, p[5][0], 1])
            diag_val = p_diag * lmbda_fix * np.exp(-lmbda_fix*ptime_diff)
            mat = np.diag(diag_val)
            for j in range(len(entry_idx)):
                mat[entry_idx[j][0], entry_idx[j][1]] = p_transition[j] * lmbda[j] * np.exp(-lmbda[j]*ptime_diff)
            mat = mat / mat.sum(axis=1)[:, None]
            assert mat.sum(axis=1).all() == 1
            tm.append(mat)
        else:
            tm.append(0)
            
    return np.array(tm)