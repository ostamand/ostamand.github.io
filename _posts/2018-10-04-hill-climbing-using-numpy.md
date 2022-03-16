---
layout: post
title: "Hill Climbing using Numpy"
math: true
---

## Introduction

![agent](/assets/agent_cartpole-v0.gif)

[CartPole-v0](https://gym.openai.com/envs/CartPole-v0/) is a simple environment where the state space consists of the cart position, cart velocity, pole angle and pole velocity at tip.

Two discrete actions are available to the agent.

- 0, push cart to the left
- 1, push cart to the right

## Stochastic Policy

In the following, we will use a stochastic policy to solve the environment.

Our policy will map each state tuple (cart_position, cart_velocity, pole_angle, pole_velocity) to probabilities of taking either action 0 or 1.

Because we don't need the policy to be differentiable in order to train, any parametrized function could be used.

In this simple example, we will use a single layer neural network i.e. a set of weights multiplying the input state vector.

$$
x = [W]_{2,4}[S]_{4,1}
$$

To get probability of taking action 0 or 1 that sums up to 1, we then apply the softmax function $\sigma$ to each element of $x$.

$$
\frac{e^{x_i}}{\sum_{i}e^{x_i}}
$$

For example, let's say that, after multiplying the state space vector by our weight matrix $[W]$, we obtain:

$$
x = [1, 2]
$$

then considering that

$$
\sum_{i}e^{x_i} = e^{1} + e^{-2} = 2.72 + 7.39 = 10.1
$$

the sigmoid function applied to $x$ will yield

$$
\sigma(x) = [e^{1}/10.1,e^{2}/10.1]\\
= [0.27,0.73]
$$

which sums up to 1.0 as needed. In this case, an agent following this policy would, with a 73% chance, take action 0 and push the cart to the left.

In summary, our stochastic policy $\pi$ is given by:

$$
\pi = \sigma([W][S])
$$

This policy can be implemented using in Python using a class:

```python
class Agent():
    """Stochastic Policy.
    One layer neural network with softmax activation function.

    """
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.w = 1e-4 * np.random.rand(state_size, action_size)

    def forward(self, state):
        x = np.dot(state, self.w)
        return np.exp(x)/sum(np.exp(x))

    def act(self, state):
        probs = self.forward(state)
        return np.random.choice(self.action_size, p=probs)
```

## Training

The policy will be trained using hill climbing with adaptive noise scaling (a kind of black box optimization where we don't need to calculate the gradients). Before we go into the details, a simple reminder. The discounted return ($G$) is defined as:

$$
G = \sum_{k=0}^{\infty} \gamma^k R_{k+1}
$$

The steps are as follow:

1. Initialize the weigths randomly with small numbers (so that initially our stochastic policy yields a 50% chance of taking either action 0 or 1).
2. Play an episode with the policy and calculate the return $G$.
3. Add noise to the best policy weights.
4. Play an episode with the modified policy and calculate $G$ .
   - If $G$ is greater than the previous best than:
     - set the best policy weights to the one of the modified policy
     - divide the noise scale by a factor of 2
   - Else:
     - multiply the noise scale by a factor of 2
5. Repeat step 3. and 4. until convergence.

Implemented in Python the training algorithm is:

```python

def train_hill_climbing(env, agent,
                        episodes=10, steps=1000, gamma=0.99, noise_scale=1e-2):
    """ Hill Climbing (Black box optimization)

    """
    scores = deque(maxlen=100)
    best_G = -np.Inf
    best_w = agent.w

    with trange(episodes) as t:
        for ep_i in t:
            state = env.reset()
            rewards = []
            for step_i in range(steps):
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                rewards.append(reward)
                state = next_state
                if done:
                    break
            scores.append(sum(rewards))

            # calculate discounted return
            G = sum([reward * gamma**k for k, reward in enumerate(rewards)])

            # found better weights
            if G >= best_G:
                best_G = G
                best_w = agent.w
                # divide noise scale by 2 with 1e-3 as minimum
                noise_scale = max(1e-3,noise_scale/2)
            # did not found better weights
            else:
                # multiply noise by 2 with 2.0 as maximum
                noise_scale = min(2, noise_scale*2)

            agent.w = best_w + noise_scale * np.random.rand(*agent.w.shape)

            current_score = np.mean(scores)
            if current_score >= 195.0:
                print(f'Env. solved in {ep_i+1-100} episodes.')
                agent.w = best_w
                break

            t.set_postfix(noise=noise_scale, Score=current_score)

```

## Results

After training, I found the following weights.

$$
[W] =
\begin{bmatrix}
    8.52 & 7.95   \\
    9.14 & 10.48  \\
    9.68 & 12.51  \\
    8.81 & 14.59
\end{bmatrix}
$$

Which means that, given a state, the probability of taken either actions is calculated as follow.

$$

x_{0} = 8.52 * \text{cart_position} + 9.14 * \text{cart_velocity} + 9.68 * \text{pole_angle} + 8.81 * \text{pole_velocity} \\

x_{1} = 7.95 * \text{cart_position} + 10.48 * \text{cart_velocity} + 12.51 * \text{pole_angle} + 14.59 * \text{pole_velocity}


$$

Finally, applying the softmax function to get the probabilities.

$$
\sigma(x) = [e^{x_{0}}/(e^{x_{0}}+e^{x_{1}}),e^{x_{1}}/(e^{x_{0}}+e^{x_{1}})]\\
$$

For example, given the state

$$
[S] =
\begin{bmatrix}
    0.018  & -0.012 & 0.046 & -0.02
\end{bmatrix}
$$

then

$$
x_{0} = 8.52 * (0.018) + 9.14 * (-0.012) + 9.68 * (0.046) + 8.81 * (-0.02) = 0.31  \\
x_{1} = 7.95 * (0.018)  + 10.48 * (-0.012) + 12.51 * (0.046)  + 14.59 * (-0.02) = 0.30 \\
\sigma(x) = [e^{0.31}/(e^{0.31}+e^{0.30}),e^{0.30}/(e^{0.31}+e^{0.30})] = [0.497, 0.503]\\
$$

Therefore, for that particular state, the policy determined that either action 0 or 1 can be taken. Imagine that the pole is vertical with a minimal velocity at the tip, then, either actions will maximize expected cumulative reward.
