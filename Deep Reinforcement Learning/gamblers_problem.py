import matplotlib.pyplot as plt

probability1 = 0.25
probability2 = 0.55
gamma = 1 # There is no decay

# Rewards for each state
rewards = [0 for i in range(100)]
rewards.append(1)
#rewards[0] = -1

# Values for each state
values = [0 for i in range(101)]

# Policies for each state
policies = [0 for i in range(101)]

# Implementing Bellman's equation for Bellman backup
def bellman_update(state):
    money_to_bet = min(state, 100-state)
    
    for bet in range(money_to_bet+1):
        #Transition states
        win_state = state + bet
        loss_state = state - bet
        
        # Updated expected value (switch between probability1 & probability2 for 0.25 & 0.55 probablities respectively)
        p = probability1
        value = p * (rewards[win_state] + gamma*values[win_state]) + (1-p) * (rewards[loss_state] + gamma*values[loss_state])
        if value > values[state]:
            values[state] = value
            policies[state] = bet

# Implementing the events for the gambler
values_after_updates = []
def gambler():
    converge = False
    while not converge:
        converge = True
        for state in range(1, 100):
            current_value = values[state]
            bellman_update(state)
            if current_value != values[state]:
                converge = False
        values_after_updates.append(values)
        
if __name__=="__main__":
    gambler()
    print("The final values after convergence:") 
    print(values_after_updates[-1])
    plt.plot(policies)