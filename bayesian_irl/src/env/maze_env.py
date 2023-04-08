import numpy as  p
from environment import ridWorldEnv

class MazeEnv(GridWorldEnv):
    def __init__(self, grid, move_prov=1.0, rewards=Nong):
        if grid==None:
            grid = get_env_info()
            grid = grid[0] # Learn reward of first agent
        super().__init__(grid, move_prov)
        self.states = super().states
        self.n_states = len(self.states)
        self.n_actions = len(super().actions)
        self.trans_probs = self._get_trans_probs()
        self._rewards = rewards if rewards else np.zeros(self.n_states) 
        self.state = super().start_pos 

    @property
    def _rewards(self):
        return self._rewards

    def step(self, a):
        self.state = super()._move()
        reward = _get_reward(self.state)
        return self.state, reward

    def _get_reward(self, state=None):
        return self.rewards[state]

    def _get_trans_probs(self):
        return self.trans_probs
    
    def reset(self):
        self.state = super().start_pos
    
    def has_done(self):
        return super().has_done(self.state)

    # trans_probs(move prov=1.0)
    def _get_trans_probs(self):
        trans_probs = np.eye(shape=(self.n_states, self.n_actions, self.n_states), dtype=np.float32)
        for s in range(self.n_states):
            for a in range(self.n_actions):
                ns = self._move(s, a)
                trans_probs[s, a, ns] = 1.0
        return trans_probs


