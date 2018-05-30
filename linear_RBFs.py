import numpy as np
from cartesian import cartesian2
import os


class LinearRBFs:
    def __init__(self, load_trained_parameters=False, e=1.0, alpha=0.4):
        self.state_list, self.dev = self.build_state_list()  # List of possible states
        self.state_list_len = len(self.state_list)
        self.number_of_states = 4
        self.rul = np.zeros((self.state_list_len, 15))
        self.membership_mult_norm = 0
        self.e = e  # Error assumption
        self.alpha = alpha  # Learning rate

        self.save_loc = 'RBF_weights/'
        if not os.path.exists(self.save_loc):
            os.makedirs(self.save_loc)

        if load_trained_parameters:
            self.load_params()
        else:
            self.wP = np.zeros((self.state_list_len, 1))  # Weights for each state
            self.wH = np.zeros((self.state_list_len, 1))  # Weights for human corrections

    def new_episode(self):
        self.rul = np.zeros((self.state_list_len, 15))

    def build_state_list(self):
        x_states = np.linspace(-1.44, 1.44, 4)
        x_dot_states = np.linspace(-0.12, 0.12, 4)
        angle_states = np.linspace(-0.126, 0.126, 4)
        angle_dot_states = np.linspace(-0.4, 0.4, 4)
        state_list = cartesian2(x_states, x_dot_states, angle_states, angle_dot_states)
        dev = np.array([x_states[1] - x_states[0], x_dot_states[1]
                        - x_dot_states[0], angle_states[1] - angle_states[0],
                        angle_dot_states[1] - angle_dot_states[0]])
        return state_list, dev

    def action(self, observation):
        membership_mult = np.zeros(self.state_list_len)
        for state in range(self.state_list_len):  # Go through all possible combinations of states
            mult = 1
            for state_index in range(self.number_of_states):  # Go through the index of the size of the state vector
                membership = np.exp((-0.5*(np.asscalar(observation[state_index]) -
                                           self.state_list[state, state_index])**2)
                                    / (self.dev[state_index]**2)) + 1e-10  # Checking membership
                mult = mult * membership
            membership_mult[state] = mult  # Saving multiplication for each state

        total_membership_mult = np.sum(membership_mult)
        self.membership_mult_norm = membership_mult / total_membership_mult

        action = np.array([np.sum(self.membership_mult_norm * np.reshape(self.wP, self.state_list_len))])
        action = np.clip(action, -1, 1)

        return action

    def update(self, h, observation):
        self.rul = np.concatenate((self.rul[:, 1:15], np.transpose(np.asmatrix(self.membership_mult_norm))), 1)
        ct = np.concatenate((np.ones((1, 10)), np.zeros((1, 5))), 1)/10
        credit_feat = np.matmul(self.rul, np.transpose(ct))
        H = np.transpose(self.wH) * credit_feat
        self.wH = self.wH + self.alpha * credit_feat * (h - H)  # Update human correction model
        H = np.transpose(self.wH) * credit_feat
        alphaP = abs(H) + 0.05

        if alphaP > 1:
            alphaP = np.array([1])
        error = h * self.e
        P = self.action(observation)

        if (P + error) > 2:
            error = 2 - P

        if (P + error) < -2:
            error = -2 - P
        self.wP = self.wP + np.asscalar(alphaP) * credit_feat * error  # Update policy
        self.wP = np.array(self.wP)

    def save_params(self):
        np.save(self.save_loc + 'wP', self.wP)
        np.save(self.save_loc + 'wH', self.wH)

    def load_params(self):
        try:
            self.wP = np.load(self.save_loc + 'wP.npy')
            self.wH = np.load(self.save_loc + 'wH.npy')
        except NameError:
            raise
