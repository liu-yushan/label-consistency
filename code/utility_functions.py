import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import cm

from matrices import start_prob, expert_emission_matrix, hmt_start_prob, hmt_expert_emission_matrix


def viterbi(observed_seq, transition_matrices, n_states=5):
    """
    Calculate the most probable sequence of hidden states using time-dependent transition matrices.
    """
    startprob = start_prob()
    emissionmat = expert_emission_matrix()
    
    N = len(observed_seq)
    Z = np.zeros(N)
    T1 = np.zeros([n_states, N])
    T2 = np.full((n_states, N), -1)

    T1[:, 0] = np.log(startprob * emissionmat[:, observed_seq[0]])
    
    for j in range(1, N):
        for i in range(n_states):
            probabilities = (T1[:, j-1] + np.log(transition_matrices[j-1][:, i])
                             + np.log(emissionmat[i, observed_seq[j]]))
            T1[i, j] = np.max(probabilities)
            T2[i, j] = np.argmax(probabilities)
        
    Z[-1] = np.argmax(T1[:, N-1])
    for i in range(N-2, 0, -1):
        Z[i] = T2[int(Z[i+1]), i+1]
    Z = Z.astype(int)
    
    return Z, T1, T2


def hmt_viterbi(X_obs, tree_obs, transition_matrices, dev_abb, dev_map, n_states=10):
    """
    Calculate the most probable hidden tree using time-dependent transition matrices.
    """
    startprob = hmt_start_prob()
    emission = hmt_expert_emission_matrix()
    tm = transition_matrices
    
    M = np.zeros((len(X_obs.index), n_states))
    opt_states = np.full((len(X_obs.index), n_states), -1)
    
    # Initialization
    for leaf in tree_obs['leaves']:
        M[int(leaf)] = np.log(emission[:, dev_map[X_obs.loc[leaf]['label']]])
    
    branches = ['branch S6-S5', 'branch S4-S3', 'branch S2-S1', 'branch S0-S1']
    for b in branches:
        branch = tree_obs[b]
        for i in range(1, len(branch)):
            m_max = []
            for j in range(n_states):
                prob = M[int(branch[i-1])] + np.log(tm[int(branch[i-1])][j])
                m_max.append(np.max(prob))
                opt_states[int(branch[i-1]), j] = np.argmax(prob)
            m_max = np.array(m_max)
            M[int(branch[i])] = m_max + np.log(emission[:, dev_map[X_obs.loc[branch[i]]['label']]])
    
    # Branching point S1 '963'
    branching_point = '963'
    child = tree_obs['children'][branching_point]
    m_max1 = []
    for j in range(n_states):
        prob = M[int(child[0])] + np.log(tm[int(child[0])][j])
        m_max1.append(np.max(prob))
        opt_states[int(child[0]), j] = np.argmax(prob)
    m_max1 = np.array(m_max1)
    m_max2 = []
    for j in range(n_states):
        prob = M[int(child[1])] + np.log(tm[int(child[1])][j])
        m_max2.append(np.max(prob))
        opt_states[int(child[1]), j] = np.argmax(prob)
    m_max2 = np.array(m_max2)
    M[int(branching_point)] = m_max1 + m_max2 + np.log(emission[:, dev_map[X_obs.loc[branching_point]['label']]])

    # Branch S1-S3
    branch = tree_obs['branch S1-S3']
    for i in range(1, len(branch)):
        m_max = []
        for j in range(n_states):
            prob = M[int(branch[i-1])] + np.log(tm[int(branch[i-1])][j])
            m_max.append(np.max(prob))
            opt_states[int(branch[i-1]), j] = np.argmax(prob)
        m_max = np.array(m_max)
        M[int(branch[i])] = m_max + np.log(emission[:, dev_map[X_obs.loc[branch[i]]['label']]])
        
    # Branching point S3 '509'
    branching_point = '509'
    child = tree_obs['children'][branching_point]
    m_max1 = []
    for j in range(n_states):
        prob = M[int(child[0])] + np.log(tm[int(child[0])][j])
        m_max1.append(np.max(prob))
        opt_states[int(child[0]), j] = np.argmax(prob)
    m_max1 = np.array(m_max1)
    m_max2 = []
    for j in range(n_states):
        prob = M[int(child[1])] + np.log(tm[int(child[1])][j])
        m_max2.append(np.max(prob))
        opt_states[int(child[1]), j] = np.argmax(prob)
    m_max2 = np.array(m_max2)
    M[int(branching_point)] = m_max1 + m_max2 + np.log(emission[:, dev_map[X_obs.loc[branching_point]['label']]])
    
    # Branch S5-S3
    branch = tree_obs['branch S5-S3']
    for i in range(1, len(branch)):
        m_max = []
        for j in range(n_states):
            prob = M[int(branch[i-1])] + np.log(tm[int(branch[i-1])][j])
            m_max.append(np.max(prob))
            opt_states[int(branch[i-1]), j] = np.argmax(prob)
        m_max = np.array(m_max)
        M[int(branch[i])] = m_max + np.log(emission[:, dev_map[X_obs.loc[branch[i]]['label']]])
            
    # Branching point S5 '1194'
    branching_point = '1194'
    child = tree_obs['children'][branching_point]
    m_max1 = []
    for j in range(n_states):
        prob = M[int(child[0])] + np.log(tm[int(child[0])][j])
        m_max1.append(np.max(prob))
        opt_states[int(child[0]), j] = np.argmax(prob)
    m_max1 = np.array(m_max1)
    m_max2 = []
    for j in range(n_states):
        prob = M[int(child[1])] + np.log(tm[int(child[1])][j])
        m_max2.append(np.max(prob))
        opt_states[int(child[1]), j] = np.argmax(prob)
    m_max2 = np.array(m_max2)
    M[int(branching_point)] = m_max1 + m_max2 + np.log(emission[:, dev_map[X_obs.loc[branching_point]['label']]])
    
    # Branch S7-S5
    branch = tree_obs['branch S7-S5']
    for i in range(1, len(branch)):
        m_max = []
        for j in range(n_states):
            prob = M[int(branch[i-1])] + np.log(tm[int(branch[i-1])][j])
            m_max.append(np.max(prob))
            opt_states[int(branch[i-1]), j] = np.argmax(prob)
        m_max = np.array(m_max)
        M[int(branch[i])] = m_max + np.log(emission[:, dev_map[X_obs.loc[branch[i]]['label']]])
    
    X_obs['Z'] = ''
    X_obs = X_obs[['label', 'Z', 'branch_id_alias', 'S7_pseudotime', 'branch_id', 'node', 'branch_lam', 'branch_dist', 
               'S0_pseudotime', 'S3_pseudotime', 'S2_pseudotime', 'S1_pseudotime', 'S6_pseudotime', 'S4_pseudotime',
               'S5_pseudotime','label_color']]
    
    root = tree_obs['root']
    Z1 = np.argmax(M[int(root)] + np.log(startprob))
    X_obs.at[root, 'Z'] = dev_abb[Z1]
    
    branch = tree_obs['branch S7-S5']
    for i in range(len(tree_obs['branch S7-S5'])-2, -1, -1):
        node = branch[i]
        X_obs.at[node, 'Z'] = dev_abb[opt_states[int(node), Z1]]
        Z1 = opt_states[int(node), Z1]
        
    branches = ['branch S6-S5', 'branch S5-S3', 'branch S4-S3', 'branch S1-S3', 'branch S2-S1', 'branch S0-S1']
    for b in branches:
        branch = tree_obs[b]
        parent = tree_obs['parents'][branch[-1]]
        Z1 = dev_map[X_obs['Z'][int(parent)]]
        X_obs.at[branch[-1], 'Z'] = dev_abb[opt_states[int(branch[-1]), Z1]]
        Z1 = opt_states[int(branch[-1]), Z1]
        for i in range(len(branch)-2, -1, -1):
            node = branch[i]
            X_obs.at[node, 'Z'] = dev_abb[opt_states[int(node), Z1]]
            Z1 = opt_states[int(node), Z1]
        
    return M, X_obs


def acc(observed_seq, hidden_seq):
    """
    Compute the accuracy of the hidden labels.
    """
    return (hidden_seq == observed_seq).mean()
    

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    Plot the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    
    Source: http://scikit-learn.org/stable/auto_examples/model_selection/
                   plot_confusion_matrix.html#confusion-matrix
    """
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis]+1e-100)
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    cbar = plt.colorbar(fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=14)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0, fontsize=14)
    plt.yticks(tick_marks, classes, fontsize=14)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize=14,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.xlabel('Expert label', fontsize=14)
    plt.ylabel('Hidden label', fontsize=14)


def transition_times(ptime_sorted, hidden_seq):
    """
    Find the pseudotimes at the class changes according to the hidden labels.
    Compute the transition times as the means of the pseudotimes of the two 
    neighboring cells at the class change.
    """
    M = hidden_seq[:-1] + hidden_seq[1:]
    ttimes = []
    for i in range(1, 9, 2):
        idx = np.where(M == i)[0][0]
        ttimes.append(np.round((ptime_sorted[idx] + ptime_sorted[idx+1]) / 2, 6))
    return ttimes


def plot_compare_labels(ptime_sorted, observed_seq, hidden_seq, transition_times):
    """
    Plot observed labels against pseudotime in a scatter plot.
    Color the consistent labels as green and the inconsistent labels as red.
    Display the transition times.
    """
    plt.figure(figsize=(10,10))
    plt.plot(ptime_sorted, hidden_seq, c='silver');
    correct = observed_seq[np.where(hidden_seq == observed_seq)]
    wrong = observed_seq[np.where(hidden_seq != observed_seq)]
    plt.scatter(ptime_sorted[np.where(hidden_seq == observed_seq)], correct, s=2, c='green');
    plt.scatter(ptime_sorted[np.where(hidden_seq != observed_seq)], wrong, s=2, c='red');
    plt.text(transition_times[0], 0.5, str(transition_times[0]));
    plt.text(transition_times[1], 1.5, str(transition_times[1]));
    plt.text(transition_times[2], 2.5, str(transition_times[2]));
    plt.text(transition_times[3], 3.5, str(transition_times[3]));
    plt.xlabel('Pseudotime')
    plt.ylabel('Expert label')
    red_dot = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                              markersize=5, alpha=0.8, label='Potentially wrong label')
    green_dot = mlines.Line2D([], [], color='green', marker='o', linestyle='None',
                              markersize=5, label='Correct label')
    plt.legend(handles=[green_dot, red_dot], prop={'size': 12})
    plt.yticks([0,1,2,3,4], labels=['PMY', 'MY', 'MMY', 'BNE', 'SNE'])
    plt.show()
    

def boxplot_ptime(ptime, labels):
    """
    Plot a boxplot of the pseudotimes, grouped by class. 
    """
    plt.figure()
    plt.boxplot(ptime[np.where(labels == 0)], positions=[1], widths=1);
    plt.boxplot(ptime[np.where(labels == 1)], positions=[3], widths=1);
    plt.boxplot(ptime[np.where(labels == 2)], positions=[5], widths=1);
    plt.boxplot(ptime[np.where(labels == 3)], positions=[7], widths=1);
    plt.boxplot(ptime[np.where(labels == 4)], positions=[9], widths=1);
    plt.xlim(0, 10)
    plt.xticks([1, 3, 5, 7, 9], labels=['PMY', 'MY', 'MMY', 'BNE', 'SNE'])
    plt.ylabel('Pseudotime')
    plt.show()
    
    
def hmt_boxplot_ptime(X_adata, dev_abb):
    """
    Plot a boxplot of the pseudotimes, grouped by class. 
    """
    for i in range(len(dev_abb)):
        plt.boxplot(np.array(X_adata.obs.loc[X_adata.obs['label'] == dev_abb[i]].S7_pseudotime), 
                    positions=[2*i+1], widths=1);
    plt.xlim(0, 20)
    plt.xticks([1, 3, 5, 7, 9, 11, 13, 15, 17, 19], labels=dev_abb, fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Labels', fontsize=14)
    plt.ylabel('Pseudotime', fontsize=14)
    plt.show()