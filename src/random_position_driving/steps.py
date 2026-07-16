import time
import numpy as np

from gym_liftoff.envs.liftoff_env import Liftoff

env = Liftoff(max_episode_time=[0, 0, 0, 30])

NUM_EPISODES = 20

total_steps = 0

action = [1024, 1024, 1024, 1024]

start_time = time.perf_counter()

for episode in range(NUM_EPISODES):

    _, info = env.reset()

    done = False

    while not done:

        _, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        total_steps += 1

end_time = time.perf_counter()

elapsed = end_time - start_time

print(f"Total steps: {total_steps}")
print(f"Total time: {elapsed:.3f} s")
print(f"Steps/s: {total_steps / elapsed:.2f}")
print(f"ms/step: {(elapsed / total_steps) * 1000:.3f}")
