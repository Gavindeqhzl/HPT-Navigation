from crowd_sim.envs.utils.agent import Agent
from crowd_sim.envs.utils.state import JointState
import torch

class Robot(Agent):
    def __init__(self, config, section):
        super().__init__(config, section)

    def act(self, ob):
        if self.policy is None:
            raise AttributeError('Policy attribute has to be set!')
        state = JointState(self.get_full_state(), ob)
        actionxy = self.policy.predict(state)

        action = torch.Tensor(list(actionxy))
        return actionxy, action

    def act_sac(self, state, noise):
        actionxy, action = self.policy.act(state, noise)
        return actionxy, action

    def act_cql(self, state, device):
        actionxy, action = self.policy.act(state, device)
        return actionxy, action
