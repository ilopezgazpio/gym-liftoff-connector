import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='liftoff-v0',
    entry_point       = 'gym_liftoff.envs:Liftoff',
    max_episode_steps = 1000000000,
    reward_threshold  = 1.0,
    nondeterministic  = False,
)
