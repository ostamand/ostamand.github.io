---
layout: post
title:  "Hill Climbing and PyTorch"
categories: rl
math: true
---
In the following, we will use a stochastic policy to solve OpenAI Gym's [CartPole-v0](https://gym.openai.com/envs/CartPole-v0/) environment.

The policy will be trained using hill climbing (black box optimization).

The state space consists of the cart position, cart velocity, pole angle and pole velocity at tip.

Two discrete actions are available.

- 0, push cart to the left 
- 1, push cart to the right 

Our policy will map each state to a probability of taking either action 0 or 1. 

$$
[A]_{2,1} = [W]_{2,4}[S]_{4,1}
$$

To get probabilty of taking action 0 or 1 that sums up to 1, we will apply the softmax function $\sigma$ to $[A]$

$$
\frac{e^{x_i}}{\sum_{i}e^{x_i}}
$$

For example, let's say that after multiplying the state space by our weights we obtain:

$$
[A] = [1, -2]
$$

then applying the sigmoid function will yield

$$
\sum_{i}e^{x_i} = e^{1} + e^{-2} = 2.71 + 0.14 = 2.85 \\
\sigma([A]) = [e^{1}/2.85,e^{-2}/2.85]\\
= [0.95,0.05]
$$

In summary, our stochatic policy $\pi$ is:

$$
\pi = \sigma([W][S])
$$





