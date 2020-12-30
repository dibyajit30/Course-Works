# -*- coding: utf-8 -*-
from __future__ import print_function

from datetime import datetime
import random
import numpy as np
import gym
import json
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.optim as optim
from utils import *
from torch import argmax
from dqn import *
import matplotlib.pyplot as plt

def gather(values, index):
    selections = [value[index[i]].item() for i, value in enumerate(values)]
    return Tensor(selections)

def optimize():
    batch_size = 32
    gamma = 0.99
    if batch_size <= len(memory):
        mini_batch = memory.sample(batch_size)
        next_states, states, actions, rewards = [], [], [], []
        for transition in mini_batch:
            next_states.append(transition.next_state)
            states.append(transition.state)
            actions.append(transition.action)
            rewards.append(transition.reward)
        next_states = Tensor(next_states)
        states = Tensor(states)
        actions = Tensor(actions).int()
        rewards = Tensor(rewards)
        
        qvalues = policy_agent(states.unsqueeze(1))
        qvalues = gather(qvalues, actions)
        next_state_qvalues = []
        for state in next_states:
            if state is None:
                next_state_qvalues.append(Tensor(0.0))
            else:
                value = target_agent(state.unsqueeze(0).unsqueeze(0)).max()
                next_state_qvalues.append(value)
        next_state_qvalues = torch.stack(next_state_qvalues, dim=0)
        expected_qvalues = rewards + (next_state_qvalues*gamma)
        
        optimizer.zero_grad()
        loss = F.smooth_l1_loss(expected_qvalues, qvalues)
        model_losses.append(loss)
        loss.backward()
        optimizer.step()

def decay_epsilon():
    global total_steps
    total_steps += 1
    if total_steps > 8000:
        return 0.1
    else:
        return 0.9 - ((total_steps/1000)*0.1)

def get_action(state):
    epsilon = decay_epsilon()
    random_value = random.random()
    if random_value > epsilon:
        return policy_agent(Tensor(state).unsqueeze(0).unsqueeze(0))
    else:
        return Tensor(np.random.rand(5))

def run_episode(env, rendering=True, max_timesteps=1000):
    
    episode_reward = 0
    step = 0
    state = env.reset()
    state = rgb2gray(state)    
    while True:
       
        action = get_action(state)
        action_mapped = output_to_action(action)
            
        next_state, r, done, info = env.step(action_mapped)   
        episode_reward += r  
        next_state = rgb2gray(next_state)
        memory.add(state, argmax(action), r, next_state)
        
        state = next_state
        step += 1
        optimize()
        
        if rendering:
            env.render()

        if done or step > max_timesteps: 
            break

    return episode_reward


if __name__ == "__main__":

    rendering = True                      
    n_episodes = 25

    target_agent = Model()
    policy_agent = Model()
    target_agent.load_state_dict(policy_agent.state_dict())
    target_agent.eval()
    optimizer = optim.RMSprop(policy_agent.parameters())
    memory_capacity = n_episodes * 200
    memory = ReplayMemory(10000)

    env = gym.make('CarRacing-v0').unwrapped
    total_steps = 0

    episode_rewards = []
    model_losses = []
    update_after_episode = 4
    for i in range(n_episodes):
        print("Running episode: ", i)
        episode_reward = run_episode(env, rendering=rendering)
        env.close()
        episode_rewards.append(episode_reward)
        if i%update_after_episode == 0:
            target_agent.load_state_dict(policy_agent.state_dict())

    results = dict()
    results["episode_rewards"] = episode_rewards
    results["mean"] = np.array(episode_rewards).mean()
    results["std"] = np.array(episode_rewards).std()
    
    plt.plot(episode_rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.savefig("episode-rewards.png")
    plt.close()
    
    plt.plot(model_losses)
    plt.title("Model losses")
    plt.savefig("model-losses.png")
    plt.close()
            
    print('... finished')