import gymnasium as gym
import numpy as np
import time
import logging
from gym_liftoff.envs.liftoff_wrappers import LiftoffWrapStability, LiftoffFloatActions

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class LiftoffReplayer:

    def __init__(self, action_path):
        self.actions = np.load(action_path)
        self.env = self.init_env()
        logger.info(f"Replayer initialized with {len(self.actions)} steps.")

    def init_env(self):
        """Initializes the specialized Liftoff Gym environment."""
        logger.info("Connecting to Liftoff Simulator via Steam...")
        env = gym.make('gym_liftoff:liftoff-v0')

        # Apply your specific wrappers
        env = LiftoffWrapStability(env)
        env = LiftoffFloatActions(env)

        return env

    def run_replay(self, skip_factor=1):
        """
        Plays back the recorded actions.
        skip_factor: Adjust this if the recording frequency exceeds the sim frequency.
        """
        logger.info("Ready. Open Liftoff and enter a flight. Press ENTER here to start.")
        input()
        logger.info("Starting in 5 seconds... Focus the Liftoff window now!")
        time.sleep(5)

        obs, _ = self.env.reset()
        done = False
        step_idx = 0

        try:
            while step_idx < len(self.actions) and not done:
                # Get action from pre-processed buffer
                action = self.actions[step_idx]

                # Step the simulator
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # Increment by skip factor
                step_idx += skip_factor

                if step_idx % 500 == 0:
                    logger.info(f"Progress: {step_idx}/{len(self.actions)} | Reward: {reward:.4f}")

        except KeyboardInterrupt:
            logger.info("Replay interrupted by user.")
        finally:
            self.env.close()
            logger.info("Environment closed.")

if __name__ == '__main__':
    # Usage:
    # 1. Run process_flight_data.py first to generate the .npy file
    # 2. Run this script
    replayer = LiftoffReplayer('./data/processed_actions.npy')

    # Try skip_factor=1 first. If the drone is too slow, increase it slightly.
    replayer.run_replay(skip_factor=1)
