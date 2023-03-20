import numpy as np
import math
import random
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score
import warnings
warnings.filterwarnings("ignore")

# Generates num_samples number of samples in 2D space within the given lower_bound and upper_bound
def generate_unlabeled_samples(num_samples, lower_bound, upper_bound):
    samples = np.random.uniform(
        lower_bound, upper_bound, size=(num_samples, 2))
    return samples

# Generate initial set of unlabelled and labelled samples
unlabelled_samples = generate_unlabeled_samples(100, [-4, -4], [4, 4])
init_labeled_samples = generate_unlabeled_samples(20, [-0.5, -0.5], [0.5,0.5])
labeled_samples = init_labeled_samples
u = random.uniform(0, 2)

def dynamics_func(state,u):
# non-linear discrete-time dynamic function
    x1,x2 = state
    x1_new = x1 + 0.05*(-x2 + 0.5*(1+x1)*u)
    x2_new = x2 + 0.05*(x1 + 0.5*(1-4*x2)*u)
    return [x1_new,x2_new]


# Function to calculate the Shannon entropy of a given sample 
def calculate_JE(sample, labeled_samples):
    gmm = GaussianMixture(n_components=2).fit(labeled_samples)
    # predict the probability of the sample belonging to each component of the GMM
    probabilities = gmm.predict_proba([sample])
    # normalize the probabilities to get the probability distribution
    prob_dist = probabilities / probabilities.sum()
    # calculate Shannon entropy of the probability distribution
    shannon_entropy = entropy(prob_dist[0])
    J = shannon_entropy
    return J

def calculate_JD(sample, labeled_samples):
    
    # calculate mutual information between the sample and labeled samples
    mutual_info = mutual_info_score(sample, labeled_samples[0])
    J = mutual_info
    return J

# Function to implement the active learning algorithm
def greedy_active_learning(labeled_samples, unlabelled_samples,batch_size):
    # Print the initial labeled samples
    print("initial labeled samples:",labeled_samples)
    # Initialize an empty list to store J values
    J_values = []
    

    for k in range(0,batch_size):
        for i in range(0,len(unlabelled_samples)):
            # Get the sample from the unlabelled samples
            sample = unlabelled_samples[i]
            # Append the J value for this sample to the J_values list
            J_values.append(calculate_JE(sample, labeled_samples)+calculate_JD(sample,labeled_samples))
            if not J_values:
                break
        
        
        # Find the argmax of J as given on paper
        idx = J_values.index(max(J_values))
        print("index of the most informative sample on unlaballed set:",idx)
        # Get the new sample from the unlabelled samples
        new_sample = unlabelled_samples[idx]
        # Append the new sample to the labeled samples
        labeled_samples = np.append(labeled_samples, [new_sample], axis=0)
        # Remove the labeled sample from the unlabelled samples
        unlabelled_samples = np.delete(unlabelled_samples, idx, axis=0)
        # Clear the J_values list for the next iteration
        J_values.clear()
        
    return labeled_samples

def stochastic_greedy_active_learning(labeled_samples, unlabelled_samples,batch_size,epsilon):
    # Print the initial labeled samples
    print("initial labeled samples:",labeled_samples)
    # Initialize an empty list to store J values
    J_values = []
    # Initialize number of samples to be selected
    q = int((len(unlabelled_samples) / batch_size) * math.log(1 / epsilon))
    
    for k in range(0,batch_size):
        unlabelled_samples_Q = []
        unlabelled_samples_Q = random.sample(list(unlabelled_samples), q)
        for i in range(0,len(unlabelled_samples_Q)):
            # Get the sample from the unlabelled samples
            sample = unlabelled_samples_Q[i]
            # Append the J value for this sample to the J_values list
            J_values.append(calculate_JE(sample, labeled_samples)+calculate_JD(sample,labeled_samples))
            if not J_values:
                break
        
        
        # Find the argmax of J as given on paper
        idx = J_values.index(max(J_values))
        print("index of the most informative sample on unlaballed set:",idx)
        # Get the new sample from the unlabelled samples
        new_sample = unlabelled_samples[idx]
        # Append the new sample to the labeled samples
        labeled_samples = np.append(labeled_samples, [new_sample], axis=0)
        # Remove the labeled sample from the unlabelled samples
        unlabelled_samples = np.delete(unlabelled_samples, idx, axis=0)
        # Clear the J_values list for the next iteration
        J_values.clear()
        
    return labeled_samples


batch_size = 10
#call the function to estimate the reachable set assuming the initally labelled samples are in the reachable set
greedy_labeled_samples = greedy_active_learning(labeled_samples, unlabelled_samples,batch_size)
#call stochastic greedy algorithm
epsilon = 0.1
stochastic_labeled_samples = stochastic_greedy_active_learning(labeled_samples, unlabelled_samples,batch_size,epsilon)
#print the labeled samples
print(*labeled_samples, sep = '\n')

#plot the samples
import matplotlib.pyplot as plt

x_labeled = [point[0] for point in greedy_labeled_samples]
y_labeled = [point[1] for point in greedy_labeled_samples]
x_stochastic =[point[0] for point in stochastic_labeled_samples]
y_stochastic = [point[1] for point in stochastic_labeled_samples]
x_unlabelled = [point[0] for point in unlabelled_samples]
y_unlabelled = [point[1] for point in unlabelled_samples]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.scatter(x_labeled, y_labeled, color='r', marker='X',s=200, label='Labeled samples')
ax1.scatter(x_unlabelled, y_unlabelled, color='b', marker='o', label='Unlabelled samples')
ax1.set_title('Greedy Active Learning')
ax1.legend()

ax2.scatter(x_stochastic, y_stochastic, color='black', marker='X',s =200, label='stochastic labelled samples')
ax2.scatter(x_unlabelled, y_unlabelled, color='b', marker='o', label='Unlabelled samples')
ax2.set_title('Stochastic Active Learning')
ax2.legend()
plt.show()




