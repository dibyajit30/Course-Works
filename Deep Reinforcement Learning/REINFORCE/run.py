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
from torch import argmax
from reinforce import *
import matplotlib.pyplot as plt

def calculate_loss():
    gamma = 0.99
    reward = 0
    adjusted_rewards = []
    for r in values['rewards'][::-1]:
        reward = r + (gamma*reward)
        adjusted_rewards.insert(0, reward)
    adjusted_rewards = Tensor(adjusted_rewards)
    adjusted_rewards = (adjusted_rewards - adjusted_rewards.mean())/(adjusted_rewards.std())
    values['actions'] = torch.stack(values['actions'], dim=0)
    loss = torch.dot((-1*values['actions']), adjusted_rewards)
    return loss

def optimize():
    loss = calculate_loss()
    model_losses.append(loss)
    optimizer.zero_grad()
    print("Loss: ", loss)
    loss.backward()
    optimizer.step()
    values['actions'] = []
    values['rewards'] = []

def get_action(state):
    state = Tensor([state])
    output = agent(state)
    action = argmax(output)
    values['actions'].append(torch.log(output[0][action]))
    return action.item()

def run_episode(env, rendering=True, max_timesteps=10000):
    
    episode_reward = 0
    step = 0
    state = env.reset()    
    a = []
    while True:
       
        action = get_action(state)
        a.append(action)
        next_state, r, done, info = env.step(action)   
        episode_reward += r  
        state = next_state
        step += 1
        values['rewards'].append(r)
        if rendering:
            env.render()

        if done or step > max_timesteps: 
            break

    return episode_reward


if __name__ == "__main__":

    rendering = True                      
    n_episodes = 100
    agent = Model()
    optimizer = optim.Adam(agent.parameters(), lr=0.001)
    env = gym.make('CartPole-v1')
    env.seed(300)
    torch.manual_seed(300)
    values = {'actions':[], 'rewards':[]}
    episode_rewards = []
    model_losses = []
    threshold_reward = env.spec.reward_threshold
    for i in range(n_episodes):
        episode_reward = run_episode(env, rendering=rendering)
        env.close()
        optimize()
        episode_rewards.append(episode_reward)
        print("Episode: ", i ," Reward: ", episode_reward)
        

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
    plt.xlabel("Episodes")
    plt.ylabel("Model losses")
    plt.savefig("model-losses.png")
    plt.close()
        
    print('... finished')