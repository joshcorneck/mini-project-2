#%%
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


class TemporalExponentialUnivariateHawkes:
    """
    A class to simulate a temporal Hawkes process with the exponential
    excitation function. Parameters:
        - alpha, beta, lam: 
    """
    def __init__(self, alpha, beta, lam):
        self.alpha = alpha
        self.beta = beta
        self.lam = lam

    def sim_S_imm(self, u):
        """
        Compute S_k+1 for immigrant points.
        """
        return -np.log(1 - u)/self.lam

    def sim_S_desc(self, u, lam_star):
        """
        Compute S_k+1 for descendant points.
        """
        return (-np.log(1 + (self.beta * np.log(1 - u)) / 
               (lam_star + self.alpha - self.lam))/self.beta)

    def comp_lam_star(self, t_k1, t_k, lam_star):
        """
        Compute lambda^*(t_k+1)
        """
        return (self.lam + np.exp(-self.beta * (t_k1 - t_k)) * 
                (lam_star - self.lam + self.alpha))

    def simulate_hawkes(self, T):

        arrival_times = []
        arrival_count = []
        X = []
        N = 0

        u = sp.stats.uniform()
        t_cur = 0; lam_star = self.lam; imm_desc_check = True
        while t_cur < T:
            # Append the arrival times, arrival count and type
            if t_cur != 0:
                arrival_times.append(t_cur)
                N += 1
                arrival_count.append(N)
                X.append(imm_desc_check * 1)

            # Sample from uniform distribution
            u1 = u.rvs(size=1); u2 = u.rvs(size=1)

            # Sample S^i and S^d
            S_i = self.sim_S_imm(u1)
            if (u2 < 1 - np.exp(-(lam_star - self.lam + self.alpha)/self.beta)):
                S_d = self.sim_S_desc(u2, lam_star)
            else:
                S_d = np.inf
            
            # Compute t_i
            if S_i < S_d:
                imm_desc_check = True
            else:
                imm_desc_check = False 

            S = min(S_i, S_d)[0]
            print(S)
            t_old = t_cur
            t_cur += S

            # Compute lam_star
            lam_star = self.comp_lam_star(t_cur, t_old, lam_star)
        
        self.arrival_times = arrival_times
        self.arrival_count = arrival_count
        self.X = X

#%%
@np.vectorize
def exponential_term(alpha, beta, tk, t):
    return alpha * np.exp(-beta * (t - tk))

alpha = 0.1; beta = 0.9; lam = 0.5; T = 10
temp_hawkes = TemporalExponentialUnivariateHawkes(alpha, beta, lam)
temp_hawkes.simulate_hawkes(T)

t_set = np.linspace(0, T, 1000); exc_func = np.tile(lam, len(t_set))
i = 0
t = t_set[0]

arrival_times = temp_hawkes.arrival_times + [T]

for j in range(len(arrival_times)):
    while t < arrival_times[j]:
        if j != 0:
            exc_func[i] += exponential_term(
                alpha, beta, arrival_times[0:j], t).sum()
        i += 1
        t = t_set[i]
exc_func[-1] = exc_func[-2]

plt.plot(t_set, exc_func);
plt.scatter(arrival_times[0:-1], 
            np.tile(lam, len(arrival_times) - 1), 
            marker='X', 
            linewidths=0.1);
plt.show()

# %%
############################################################
############ Multivariate Hawkes process ###################
############################################################

def alphai_func(alphai, mui, phii, t):
    exp_terms = np.exp(-(mui + phii) * t)
    return (alphai + mui * exp_terms.sum())

def betaj_func(betaj, muj, phij, t):
    exp_terms = np.exp(-(muj + phij) * t)
    return (betaj + muj * exp_terms.sum())

def gammaij_func(gammai, gammaj, thetai, thetaj, 
            nui, nuj, t):
    exp_terms = np.exp(-(thetai + nui) * (thetaj + nuj) * t)
    return (np.dot(gammai, gammaj) + np.sum(nui * nuj * exp_terms))
    
def Hawkes_sampler(lam_fnc, T):
    """
    Sample a Hawkes process with a given intensity function. Parameters:
    
        - lam_fnc: a lambda function taking a single time argument, all
                   other necessary parameters are given in the lambda
                   definition.
        - T: time interval we simulate over.
    """
    t_cur = 0 # Current time
    t_events = [] # List to store event times

    while t_cur < T:
        lam_max = lam_fnc(t_cur - np.array(t_events)) # Compute right limit
        u1 = np.random.uniform(size=1)[0]  
        t_cur += -np.log(u1)/lam_max # Compute proposal waiting time and add
        u2 = np.random.uniform(size=1)[0]   
        if (u2 < lam_fnc(t_cur - np.array(t_events))/lam_max) & (t_cur < T):
            # With probability lambda(t_cur-)/lambda(t_cur+) accept
            t_events.append(t_cur)
        elif t_cur > T:
            break

    return t_events

lam_fnc = lambda t: alphai_func(lam, alpha, beta - alpha, t)
t_events = Hawkes_sampler(lam_fnc, 10)

T = 10
t_set = np.linspace(0, T, 1000); exc_func = np.tile(lam, len(t_set))
i = 0
t = t_set[0]

arrival_times = t_events + [T]

for j in range(len(arrival_times)):
    while t < arrival_times[j]:
        if j != 0:
            exc_func[i] += lam_fnc(t - arrival_times[0:j]) - lam
        i += 1
        t = t_set[i]
exc_func[-1] = exc_func[-2]

plt.plot(t_set, exc_func);
plt.scatter(arrival_times[0:-1], 
            np.tile(lam, len(arrival_times) - 1), 
            marker='X', 
            linewidths=0.1);
plt.show()

# %%
