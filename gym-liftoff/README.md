# gym-liftoff environment

Openai-gym environment to interact with Steam Lift Off Drone Simulation game.

For a complete description of the environment and its rules check the paper.

## Instalation

```
pip3 install -e gym-liftoff
```

or

```
cd gym-liftoff
pip3 install -e .
```

## Usage
create an instance of the environment with

```
import gym
env = gym.make('gym_liftoff:liftoff-v0')
env.action_space
env.action_space.n
env.action_space.sample()
env.observation_space
env.observation_space.low
env.observation_space.high
env.reset()
env.render()
env.env.state
```