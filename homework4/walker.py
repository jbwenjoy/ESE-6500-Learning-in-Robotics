from dm_control import suite, viewer
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class uth_t(nn.Module):
    # Actor-
    def __init__(s, xdim, udim, hdim=32, fixed_var=True):
        super().__init__()
        s.xdim, s.udim = xdim, udim
        s.fixed_var = fixed_var

        ### TODO

        ### END TODO

    def forward(s, x):
        ### TODO

        ### END TODO
        return mu, std


def rollout(e, uth, T=1000):
    """
    e: environment
    uth: controller
    T: time-steps
    """

    traj = []
    t = e.reset()
    x = t.observation
    x = np.array(x['orientations'].tolist() + [x['height']] + x['velocity'].tolist())
    for _ in range(T):
        with th.no_grad():
            u, _ = uth(th.from_numpy(x).float().unsqueeze(0))
        r = e.step(u.numpy())
        x = r.observation
        xp = np.array(x['orientations'].tolist() + [x['height']] + x['velocity'].tolist())

        t = dict(xp=xp, r=r.reward, u=u, d=r.last())
        traj.append(t)
        x = xp
        if r.last():
            break
    return traj


"""
Setup walker environment
"""
r0 = np.random.RandomState(42)
e = suite.load('walker', 'walk',
               task_kwargs={'random': r0})
U = e.action_spec()
udim = U.shape[0]
X = e.observation_spec()
xdim = 14 + 1 + 9

"""
#Visualize a random controller
"""
# def u(dt):
#     return np.random.uniform(low=U.minimum,
#                              high=U.maximum,
#                              size=U.shape)
# viewer.launch(e,policy=u)

# Example rollout using a network/
# uth = uth_t(xdim, udim)
# traj = rollout(e, uth)
