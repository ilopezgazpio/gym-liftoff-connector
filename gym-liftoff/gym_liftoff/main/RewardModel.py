


class RewardModel:
    def __init__(self):
        self.model = None
        self.model = self.load_model()
    
    def reward(self, state, action, info):
        """
        Calculate reward based on the state, action and info
        
        TODO

        """
        
        return 0