# Implementation-of-Iterative-Policy-Evaluation-for-a-Finite-MDP
## Aim

To implement iterative policy evaluation using Gymnasium and estimate the state-value function $V^\pi(s)$ for a fixed random policy.

---
## Software Requirements

Install the required Python packages:

```bash
pip install gymnasium numpy
```

---

## Environment Used

The experiment uses the **FrozenLake-v1** environment from Gymnasium.

FrozenLake is a grid-based reinforcement learning environment where the agent starts from a start state and tries to reach the goal state without falling into holes.

For the default 4 x 4 FrozenLake map:

| Component | Description |
|---|---|
| Observation space | 16 discrete states |
| Action space | 4 discrete actions |
| Actions | 0 = Left, 1 = Down, 2 = Right, 3 = Up |
| Reward | +1 for reaching goal, 0 otherwise |
| Terminal states | Goal and holes |

---

## Problem Statement

Evaluate a fixed random policy in the FrozenLake-v1 environment.

The agent follows a random policy, where each of the four actions is selected with equal probability:

$$
\pi(a|s) = \frac{1}{4}
$$

This probability refers to the policy's action-selection probability. The environment transition probabilities are obtained from Gymnasium using `env.P[state][action]`. If `is_slippery=True`, the agent may not move in the intended direction due to stochastic transitions.

The objective is to estimate the state-value function:

$$
V^\pi(s)
$$

---

## Theory

The state-value function under policy $pi$, denoted by $V^\pi(s)$, represents the expected return starting from state $s$ and following policy $pi$.

The Bellman expectation equation is:

```math
V^\pi(s) =
\sum_a \pi(a|s)
\sum_{s'} P(s'|s,a)
\left[
R(s,a,s') + \gamma V^\pi(s')
\right]
```

Where:

| Symbol | Meaning |
|---|---|
| $s$ | Current state |
| $a$ | Action |
| $s'$ | Next state |
| $\pi(a \mid s)$ | Probability of selecting action $a$ in state $s$ |
| $P(s' \mid s,a)$ | Transition probability |
| $R(s,a,s')$ | Reward |
| $\gamma$ | Discount factor |
| $V^\pi(s)$ | Value of state $s$ under policy $\pi$ |

---
## Algorithm

1. Create the FrozenLake-v1 environment using Gymnasium.
2. Access the transition model of the environment.
3. Initialize \(V(s)=0\) for all states.
4. Define a random policy where each action has equal probability.
5. For each state:
   - For each action:
     - Read transition probability, next state, reward, and terminal status.
     - Apply the Bellman expectation equation.
6. Repeat until the value function converges.
7. Display the final value function as a 4 x 4 grid.

---

## Program

```
import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1", is_slippery=True)
env = env.unwrapped


def policy_evaluation(env, gamma=0.5, theta=1e-8):
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # Initialize value function
    V = np.zeros(n_states)

    # Random policy
    policy = np.ones((n_states, n_actions)) / n_actions

    iterations = 0

    while True:
        delta = 0
        new_V = np.copy(V)

        for state in range(n_states):
            v = 0

            for action, action_prob in enumerate(policy[state]):
                for prob, next_state, reward, done in env.P[state][action]:
                    v += action_prob * prob * (
                        reward + gamma * V[next_state] * (not done)
                    )

            delta = max(delta, abs(v - V[state]))
            new_V[state] = v

        V = new_V
        iterations += 1

        if delta < theta:
            break

    return V, iterations

V, iterations = policy_evaluation(env)

print("Name: shruthi D N")
print("Reg.NO.: 212223240155")
print("Number of Iterations:", iterations)
print("\nState-Value Function as 4x4 Grid:\n")
print(V.reshape((4, 4)))

```

## Output

```
Name: shruthi D N
Reg.NO.: 212223240155
Number of Iterations: 17

State-Value Function as 4x4 Grid:

[[5.72968370e-05 1.14833186e-04 7.46554220e-04 1.24423499e-04]
 [2.28974510e-04 0.00000000e+00 4.98664047e-03 0.00000000e+00]
 [1.54554224e-03 1.05898347e-02 3.91465784e-02 0.00000000e+00]
 [0.00000000e+00 4.40265693e-02 2.97596162e-01 0.00000000e+00]]



```
---

## Result

Iterative policy evaluation was implemented successfully using the Gymnasium FrozenLake environment. The state-value function for the fixed random policy was estimated using the Bellman expectation equation.

---

## Inference

```
The state-value function converges after repeated application of the Bellman expectation equation. States closer to the goal have higher values, while hole states and terminal states have zero value. The random policy results in relatively low state values because actions are chosen uniformly at random, reducing the probability of reaching the goal efficiently.



```




---


