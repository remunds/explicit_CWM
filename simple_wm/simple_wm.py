# goal: implement a simple world model in torch
# requires: a dynamics model predicting the distributions of the next state s_{t+1}, reward r_t and discount \gamma_t given s_t, which is the explicit object-centric world state (ocatari)
#           a simple actor critic model that works on the imaginary trajectories generated by the dynamics model
# Q: how can we get randomness in the imaginations of the world model? 


import random
from ocatari import OCAtari
import torch
from torch import nn

# First: collect trajectories from the environment, following the actor
env = OCAtari(env_name='Pong', mode="ram", obs_mode="obj", buffer_window_size=1)
obs, info = env.reset()
state_size = obs.shape[1]

def collect_transitions(env: OCAtari, step_num=100):
    transitions = []
    for _ in range(step_num):
        action = {'action': 0, 'reset': False}
        if len(transitions) > 0 and (transitions[-1]["truncated"] or transitions[-1]["terminated"]):
            obs, info = env.reset()
            reward = 0.0
            truncated = False
            terminated = False
        else:
            obs, reward, truncated, terminated, info = env.step(action['action'])
        obs_dict = {
            "obs": torch.tensor(obs.flatten(), dtype=torch.float32),
            "reward": torch.tensor(reward, dtype=torch.float32, device=device),
            "truncated": truncated,
            "terminated": terminated,
            "info": info 
        }
        transitions.append(obs_dict)
        if truncated or terminated:
            obs, info = env.reset()
    # random.shuffle(transitions)
    return transitions

class WorldModel(nn.Module):
    def __init__(self, state_size, hidden_size, categorical_size=256):
        super(WorldModel, self).__init__()
        self.state_size = state_size
        self.input_size = state_size + 1 # (1 = action)
        self.output_size = state_size * 2# (deterministic, stochastic)
        self.hidden_size = hidden_size
        self.categorical_size = categorical_size

        # size is flattened observation
        self.dynamics = nn.GRU(input_size=self.input_size, hidden_size=self.output_size)
        #
        # describe the next-state distribution by a categorical distribution with 256 categories for each dimension
        self.dyn_params = nn.Linear(state_size, state_size * self.categorical_size)

        self.reward = nn.Sequential(
            nn.Linear(self.output_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 2),
        )
        self.discount = nn.Sequential(
            nn.Linear(self.output_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
        )
    
    def forward(self, state_seq, action):
        # dyn model: s_t ~ p(s_t | s_{t-1}, h_{t-1}, a_{t-1})
        # since we are 8-bit, we use a categorical distribution with 256 categories for each dimension

        # compute next state from state_seq
        # compute reward and discount from last state 
        # stochastic, deterministic = self.dynamics(torch.cat([state_seq, action.view(1, 1)], dim=1))
        output, _ = self.dynamics(torch.cat([state_seq, action.view(1, 1)], dim=1))
        stochastic = output[:, :self.state_size]
        deterministic = output[:, self.state_size:]
        stochastic = self.dyn_params(stochastic).view(-1, self.state_size, self.categorical_size) #(Batch, 6, 256)
        state_dist = torch.distributions.Categorical(logits=stochastic)
        sample = state_dist.sample()
        probs = state_dist.log_prob(sample)
        # straight-through estimator (to keep gradients while sampling)
        state_sample = sample + probs - probs.detach()

        #h_t, s_t
        # import ipdb; ipdb.set_trace()
        concatenated = torch.cat([deterministic, state_sample], dim=1)
        reward_params = self.reward(concatenated).view(-1, 2)
        reward_mu = reward_params[:, 0]
        reward_sigma = reward_params[:, 1].clamp(min=1e-5)
        reward_dist = torch.distributions.Normal(reward_mu, reward_sigma)

        discount_params = self.discount(concatenated).view(-1, 1)
        discount_dist = torch.distributions.Bernoulli(logits=discount_params)

        return state_dist, reward_dist, discount_dist


# train wm on transitions
def train(wm, transitions, device, num_epochs=100):
    optimizer = torch.optim.Adam(wm.parameters(), lr=1e-4)
    for epoch in range(num_epochs):
        for t in range(len(transitions) - 1):
            optimizer.zero_grad()
            s_t = transitions[t]["obs"].view(1, -1).to(device)
            s_t1 = transitions[t + 1]["obs"].view(1, -1).to(device)
            r_t = transitions[t]["reward"].to(device)
            # d_t = transitions[t]["terminated"]
            action = torch.tensor(0.0, device=device)
            # s_t1_pred, r_t_pred, d_t_pred = wm(s_t, h_0, action)
            s_t1_pred, r_t_pred, d_t_pred = wm(s_t, action)
            loss = -s_t1_pred.log_prob(s_t1).mean() - r_t_pred.log_prob(r_t)# + d_t_pred.log_prob(d_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(wm.parameters(), max_norm=5.0)
            optimizer.step()
        print(f"Epoch: {epoch}, Loss: {loss.item()}")
        if loss.item() < -5: 
            break

def predict(wm, start, device, num_steps=100):
    transitions = []
    state_seq = start
    for _ in range(num_steps):
        action = torch.tensor(0.0, device=device)
        state_dist, reward_dist, discount_dist = wm(state_seq, action)
        state_sample = state_dist.mode
        reward_sample = reward_dist.mode
        discount_sample = discount_dist.mode
        state_seq = state_sample
        transitions.append({
            "state": state_sample,
            "reward": reward_sample,
            "discount": discount_sample
        })
    return transitions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wm = WorldModel(state_size, state_size, 256).to(device)
print(wm)

transitions = collect_transitions(env, step_num=300)

train(wm, transitions, device, num_epochs=100)

start = transitions[0]["obs"].view(1, -1).to(device)
pred_transitions = predict(wm, start, device, num_steps=300)
for t in pred_transitions:
    print(t["state"].cpu().numpy())