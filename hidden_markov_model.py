from sklearn.preprocessing import MinMaxScaler
from scipy.stats import norm, tmean, tvar
import numpy as np

class State:
    def __init__(self, Dataset):
        self.initial_probability = np.random.rand()
        self.forward_probabilities = np.zeros(len(Dataset))
        self.backward_probabilities = np.zeros(len(Dataset))
        self.mean = np.random.rand()
        self.variance = np.random.rand()
        self.state_occupations = np.zeros(len(Dataset))

class Hidden_Markov_Model:
    def __init__(self, Dataset, number_of_states):
        self.number_of_states = number_of_states
        self.Dataset = Dataset
        self.States = np.array([State(Dataset) for i in range(number_of_states)])
        self.transition_matrix = np.random.rand(number_of_states, number_of_states)
        self.transition_matrix = self.transition_matrix / np.sum(self.transition_matrix, axis=1, keepdims=True)
        initial_probabilities = np.array([state.initial_probability for state in self.States])
        for state in self.States:
            state.initial_probability /= np.sum(initial_probabilities)

        self.state_transitions = np.zeros((len(Dataset)-1, number_of_states, number_of_states))
        self.forward_scales = np.zeros(len(Dataset))
        self.log_likelihood = 0

    def normalizing(self, array, t):
        normalizing_factor = 0
        for i in range(self.number_of_states):
            normalizing_factor += getattr(self.States[i], array)[t]
        if normalizing_factor > 0:
            for i in range(self.number_of_states):
                getattr(self.States[i], array)[t] = getattr(self.States[i], array)[t] / normalizing_factor
            if array == 'forward_probabilities':
                self.forward_scales[t] = 1 / normalizing_factor
        else:
            self.forward_scales[t] = 0
            for i in range(self.number_of_states):
                getattr(self.States[i], array)[t] = 0

    def forward_algorithm(self, t):
        for i in range(self.number_of_states):
            forward_sum = 0
            for j in range(self.number_of_states):
                forward_sum += self.States[j].forward_probabilities[t-1] * self.transition_matrix[j][i]
            self.States[i].forward_probabilities[t] = forward_sum * norm.pdf(self.Dataset[t], loc=self.States[i].mean, scale=np.sqrt(self.States[i].variance))
        self.normalizing('forward_probabilities', t)

    def backward_algorithm(self, t):
        for i in range(self.number_of_states):
            backward_sum = 0
            for j in range(self.number_of_states):
                backward_sum += self.transition_matrix[i][j] * norm.pdf(self.Dataset[t+1], loc=self.States[j].mean, scale=np.sqrt(self.States[j].variance)) * self.States[j].backward_probabilities[t+1]
            self.States[i].backward_probabilities[t] = backward_sum
        self.normalizing('backward_probabilities', t)

    def state_occupation(self, t):
        denominator = 0
        for j in range(self.number_of_states):
            denominator += self.States[j].forward_probabilities[t] * self.States[j].backward_probabilities[t]
        if denominator > 0:
            for i in range(self.number_of_states):
                self.States[i].state_occupations[t] = self.States[i].forward_probabilities[t] * self.States[i].backward_probabilities[t] / denominator
        else:
            for i in range(self.number_of_states):
                self.States[i].state_occupations[t] = 0

    def state_transition(self, t):
        denominator = 0
        for k in range(self.number_of_states):
            for l in range(self.number_of_states):
                denominator += self.States[k].forward_probabilities[t] * self.transition_matrix[k][l] * norm.pdf(self.Dataset[t+1], loc=self.States[l].mean, scale=np.sqrt(self.States[l].variance)) * self.States[l].backward_probabilities[t+1]
        if denominator > 0:
            for i in range(self.number_of_states):
                for j in range(self.number_of_states):
                    self.state_transitions[t][i][j] = self.States[i].forward_probabilities[t] * self.transition_matrix[i][j] * norm.pdf(self.Dataset[t+1], loc=self.States[j].mean, scale=np.sqrt(self.States[j].variance)) * self.States[j].backward_probabilities[t+1] / denominator
        else:
            for i in range(self.number_of_states):
                for j in range(self.number_of_states):
                    self.state_transitions[t][i][j] = 0

    def e_step(self):
        for i in range(self.number_of_states):
            self.States[i].forward_probabilities[0] = self.States[i].initial_probability * norm.pdf(self.Dataset[0], loc=self.States[i].mean, scale=np.sqrt(self.States[i].variance))
            self.States[i].backward_probabilities[len(self.Dataset)-1] = 1
        self.normalizing('forward_probabilities', 0)

        for t in range(1, len(self.Dataset)):
            self.forward_algorithm(t)
        for t in range(len(self.Dataset)-2, -1, -1):
            self.backward_algorithm(t)
        for t in range(len(self.Dataset)):
            self.state_occupation(t)
        for t in range(len(self.Dataset)-1):
            self.state_transition(t)

    def m_step(self):
        for i in range(self.number_of_states):
            self.States[i].initial_probability = self.States[i].state_occupations[0]

        for i in range(self.number_of_states):
            denominator = np.sum(self.States[i].state_occupations[:-1])
            if denominator > 0:
                for j in range(self.number_of_states):
                    self.transition_matrix[i][j] = np.sum(self.state_transitions[:, i, j]) / denominator
            else:
                self.transition_matrix[i][:] = 1 / self.number_of_states

        for i in range(self.number_of_states):
            denominator = np.sum(self.States[i].state_occupations)
            if denominator > 0:
                self.States[i].mean = np.sum(self.States[i].state_occupations * self.Dataset) / denominator
            else:
                self.States[i].mean = np.random.rand()

        for i in range(self.number_of_states):
            denominator = np.sum(self.States[i].state_occupations)
            if denominator > 0:
                self.States[i].variance = np.sum(self.States[i].state_occupations * (self.Dataset - self.States[i].mean) ** 2) / denominator
                if self.States[i].variance <= 0:
                    self.States[i].variance = np.random.rand()
            else:
                self.States[i].variance = np.random.rand()

    def baum_welch_algorithm(self, threshold, max_iterations=100):
        self.e_step()
        self.m_step()
        new_log_likelihood = -np.sum(np.log(self.forward_scales + 1e-100))
        difference = float('inf')
        iteration = 0

        while difference > threshold and iteration < max_iterations:
            iteration += 1
            previous_log_likelihood = new_log_likelihood
            self.e_step()
            self.m_step()
            new_log_likelihood = -np.sum(np.log(self.forward_scales + 1e-100))
            difference = np.abs(new_log_likelihood - previous_log_likelihood)
            print(f'Iteration {iteration} . . . Log-Likelihood = {new_log_likelihood}')

        self.log_likelihood = new_log_likelihood
    
    def get_parameters(self):
        for i in range(self.number_of_states):
            print(f'==================================== {i} =====================================')
            print(f'Initial Probability: {self.States[i].initial_probability}')
            print(f'Mean: {self.States[i].mean}')
            print(f'Variance: {self.States[i].variance}')
            print('\n')
        print(f'Transition Matrix: {self.transition_matrix}')
        print(f'Log Likelihood: {self.log_likelihood}')