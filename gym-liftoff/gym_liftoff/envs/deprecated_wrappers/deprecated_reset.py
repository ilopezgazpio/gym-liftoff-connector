def reset_deprecated(self, seed=None, options=None):
    super().reset(seed=seed)  # sets Gymnasium RNG
    self._has_reset = True

    if hasattr(self, "resetting") and self.resetting:
        # already called from wrapper, skip duplication
        pass
    else:
        self.resetting = True
        self.virtual_gamepad.reset()
        pyautogui.press('r')
        time.sleep(1.5)
        self.virtual_gamepad.reset()
        self.act([900, 1024, 1024, 1024], from_reset=True)
        time.sleep(1)

    self.time = 0
    self.past_action = None
    self.state = self.video_sampler.sample(region=(1280, 0, 1920, 1080))
    observation = self.observation()
    done = self.__episode_terminated__(info)
    self.crash_detector.reset()
    logger.info("Reward obtained: {}".format(reward))

    return observation, reward, done, False, info