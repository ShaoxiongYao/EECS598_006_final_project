from nmp.policy import utils
from nmp import settings
from rlkit.torch.core import eval_np
import time
import numpy as np
import torch
from nmp.model.pointnet import process_input
from rlkit.torch.pytorch_util import set_gpu_mode

set_gpu_mode(True)

policies = []
log_dir = settings.log_dir()

exp_names=['/home/yixuan/sshape_boxes_global_1024/seed1/itr_900.pkl',
'/home/yixuan/sshape_boxes_global_1024/seed1/itr_920.pkl',
'/home/yixuan/sshape_boxes_global_1024/seed1/itr_940.pkl',
'/home/yixuan/sshape_boxes_global_1024/seed1/itr_960.pkl',
'/home/yixuan/sshape_boxes_global_1024/seed1/itr_980.pkl']

cpu=False
stochastic=False
for exp_name in exp_names:
    policies.append(utils.load(log_dir, exp_name, cpu, stochastic))

# policies[0].get_action(np.ones((1,6))) == policies[0].stochastic_policy.forward(torch.ones((1,6)), deterministic=True)[0]

class policiesEnsemble(torch.nn.Module):
    def __init__(self, agents):
        super().__init__()
        self.agents = torch.nn.ModuleList()
        # self.agents = []
        for agent in agents:
            self.agents.append(agent.stochastic_policy)
    def forward(self, observations : torch.Tensor) -> torch.Tensor:
        actions=[]
        # actions_info=[]
        futures=[]
        # print("get_action_kwargs:", get_action_kwargs)
        # print("**get_action_kwargs:", **get_action_kwargs)
        kwargs={"deterministic" : True}
        for agent in self.agents:
            futures.append(torch.jit.fork(agent.forward, observations))
        for future in futures:
            single_a_torch, _, _, _, _, _, _, _ = torch.jit.wait(future)
            actions.append(single_a_torch)
            # actions_info.append({})
        return torch.stack(actions).sum(dim=0)

start_t = time.time()

for i in range(1000):
    ens = policiesEnsemble(policies) # work
    ens = torch.jit.script(policiesEnsemble(policies)) # fail
    actions = ens(torch.ones((1000,6), device='cuda'))
    # for policy in policies:
    #     policy.get_action(np.ones(6000))
# policies[0].stochastic_policy.block0(torch.ones(5000,6))
# policies[0].stochastic_policy.block0(torch.ones(5000,6))
# eval_np(policies[0].stochastic_policy, np.ones((1,6)), False)
# eval_np(policies[0].stochastic_policy, np.ones((100,6)), False)
end_t = time.time()
print(end_t-start_t)

# obs = np.ones((100,6))[None]

# obstacles, links, goal, action = process_input(
#     policies[0].stochastic_policy.input_indices,
#     policies[0].stochastic_policy.obstacle_point_dim,
#     policies[0].stochastic_policy.coordinate_frame,
#     *obs
# )

# if policies[0].stochastic_policy.coordinate_frame == "local":
#     # early action integration
#     h = torch.cat((obstacles, action), dim=2)
#     # late action integration
#     # h = obstacles
# elif policies[0].stochastic_policy.coordinate_frame == "global":
#     h = torch.cat((obstacles, links, goal, action), dim=2)

# if policies[0].stochastic_policy.coordinate_frame == "local":
#     if policies[0].stochastic_policy.goal_dim > 0:
#         h = torch.cat((h, goal), dim=1)

# h = policies[0].stochastic_policy.block1(h)

# preactivation = policies[0].stochastic_policy.last_fc(h)
# output = policies[0].stochastic_policy.output_activation(preactivation)
