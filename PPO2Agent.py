import time
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter

from our_gym import FruitPickingEnv
from PPO2Network import PPO2Network

# Initialize environment and TensorBoard writer
env = FruitPickingEnv()

# Get initial observation maps from the environment
obs_map, pos_map, goal_map = env._observe()
obs_map = torch.tensor(obs_map, dtype=torch.float32)
pos_map = torch.tensor(pos_map, dtype=torch.float32)
goal_map = torch.tensor(goal_map, dtype=torch.float32)
input_all_state = torch.stack((obs_map, pos_map, goal_map), dim=0).unsqueeze(0)

# Define observation and action dimensions based on environment
state_dim = input_all_state.shape  # Adjusted observation space dimensions
action_dim = env.action_space.n  # Adjusted action space dimensions

# Hyperparameters for PPO
RENDER = False
EP_MAX = 1000  # Max number of episodes for training
EP_LEN = 1000  # Max length of each episode
GAMMA = 0.6  # Discount factor for rewards
A_LR = 0.0001  # Learning rate for actor
C_LR = 0.0003  # Learning rate for critic
BATCH = 128  # Batch size
A_UPDATE_STEPS = 10  # Actor update steps
C_UPDATE_STEPS = 10  # Critic update steps
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5, epsilon=0.2),  # KL penalty method
    dict(name='clip', epsilon=0.2)  # Clipped surrogate objective method
][1]  # Select optimization method
Switch = 1  # Mode switch for training (1 for testing mode)
if Switch == 0:
    writer = SummaryWriter(log_dir="runs/PPO2_training")

# Define PPO2 Actor class
class PPO2Actor():
    def __init__(self):
        # Initialize two actor networks for policy updates
        self.old_pi, self.new_pi = PPO2Network(state_dim, action_dim), PPO2Network(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.new_pi.parameters(), lr=A_LR, eps=1e-5)
        self.step = 0  # Track TensorBoard step

    def choose_action(self, s):
        # Get action distribution and sample action
        mean, _, _ = self.old_pi(s)
        dist = torch.distributions.Categorical(logits=mean)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach().numpy(), action_logprob.detach().numpy()

    def update_oldpi(self):
        # Synchronize old policy with new policy after update
        self.old_pi.load_state_dict(self.new_pi.state_dict())

    def learn(self, bs, ba, adv, bap):
        # Perform policy update using PPO with clipped objective
        bs = torch.FloatTensor(bs)
        ba = torch.FloatTensor(ba)
        adv = torch.FloatTensor(adv)
        bap = torch.FloatTensor(bap)
        for _ in range(A_UPDATE_STEPS):
            mean, _, _ = self.new_pi(bs)
            dist_new = torch.distributions.Categorical(logits=mean)
            action_new_logprob = dist_new.log_prob(ba)
            ratio = torch.exp(action_new_logprob - bap.detach())
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - METHOD['epsilon'], 1 + METHOD['epsilon']) * adv
            loss = -torch.min(surr1, surr2).mean()
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.new_pi.parameters(), 0.5)
            self.optimizer.step()

            # Log policy loss to TensorBoard
            writer.add_scalar("Loss/Policy_Loss", loss.item(), self.step)
            self.step += 1


# Define PPO2 Critic class
class PPO2Critic():
    def __init__(self):
        # Initialize critic network and optimizer
        self.critic_v = PPO2Network(state_dim, 1)
        self.optimizer = torch.optim.Adam(self.critic_v.parameters(), lr=C_LR, eps=1e-5)
        self.lossfunc = nn.MSELoss()  # Mean Squared Error loss for value update
        self.step = 0

    def get_v(self, s):
        # Estimate the state value
        input_state = torch.FloatTensor(s[0]).unsqueeze(0)
        _, _, Q_value = self.critic_v(input_state)
        return Q_value

    def learn(self, bs, br):
        # Perform value update for the critic
        bs = torch.FloatTensor(bs)
        reality_v = torch.FloatTensor(br)
        desired_value = 0
        for _ in range(C_UPDATE_STEPS):
            desired_value = self.get_v(bs)
            td_e = self.lossfunc(reality_v, desired_value)
            self.optimizer.zero_grad()
            td_e.backward()
            nn.utils.clip_grad_norm_(self.critic_v.parameters(), 0.5)
            self.optimizer.step()

            # Log value loss to TensorBoard
            writer.add_scalar("Loss/Value_Loss", td_e.item(), self.step)
            self.step += 1
        return (reality_v - desired_value).detach()


# PPO2 training function
def ppo2train(env, EP_MAX, EP_LEN, BATCH, GAMMA, actor, critic, RENDER=False):
    print('PPO2 Training...')
    all_ep_r = []  # Record of all episode rewards

    for episode in range(EP_MAX):
        observation, _ = env.reset()
        buffer_s, buffer_a, buffer_r, buffer_a_logp = [], [], [], []
        reward_total = 0

        for timestep in range(EP_LEN):
            if RENDER:
                env.render()

            # Gather new observations from environment
            observation = env._observe()
            obs_map, pos_map, goal_map = observation
            obs_map = torch.tensor(obs_map, dtype=torch.float32)
            pos_map = torch.tensor(pos_map, dtype=torch.float32)
            goal_map = torch.tensor(goal_map, dtype=torch.float32)
            observation = torch.stack((obs_map, pos_map, goal_map), dim=0).unsqueeze(0)

            # Select action based on policy
            action, action_logprob = actor.choose_action(observation)
            observation_, reward, done, info = env.step(action)

            if done:
                print("finish this episode!")
                break

            obs_map_, pos_map_, goal_map_ = observation_
            obs_map_ = torch.tensor(obs_map_, dtype=torch.float32)
            pos_map_ = torch.tensor(pos_map_, dtype=torch.float32)
            goal_map_ = torch.tensor(goal_map_, dtype=torch.float32)

            # Store experience
            observation_ = torch.stack((obs_map_, pos_map_, goal_map_), dim=0).unsqueeze(0)
            buffer_s.append(observation)
            buffer_a.append(action)
            buffer_r.append(reward)
            buffer_a_logp.append(action_logprob)
            reward_total += reward

            # Update networks
            if (timestep + 1) % BATCH == 0 or timestep == EP_LEN - 1:
                v_observation_ = critic.get_v(observation_).squeeze()
                discounted_r = []
                for reward in buffer_r[::-1]:
                    v_observation_ = reward + GAMMA * v_observation_
                    discounted_r.append(v_observation_.detach().numpy())
                discounted_r.reverse()

                # Prepare experience for network update
                bs, ba, br, bap = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r), np.vstack(
                    buffer_a_logp)
                buffer_s, buffer_a, buffer_r, buffer_a_logp = [], [], [], []
                advantage = critic.learn(bs, br)
                actor.learn(bs, ba, advantage, bap)
                actor.update_oldpi()

        # Log reward per episode and save to TensorBoard
        all_ep_r.append(reward_total if episode == 0 else all_ep_r[-1] * 0.9 + reward_total * 0.1)
        print(f"\rEp: {episode} | rewards: {reward_total}")
        writer.add_scalar("Reward/Per Episode", reward_total, episode)

        # Save model periodically
        if episode == EP_MAX - 1 or (episode + 1) % 100 == 0:
            torch.save({'net': actor.old_pi.state_dict(), 'opt': actor.optimizer.state_dict(), 'i': (episode + 1)},
                       "PPO2_Models/PPO2_model_actor_{}.pth".format(episode + 1))
            torch.save({'net': critic.critic_v.state_dict(), 'opt': critic.optimizer.state_dict(), 'i': (episode + 1)},
                       "PPO2_Models/PPO2_model_critic_{}.pth".format(episode + 1))

    env.close()
    writer.close()

def ppo2test(env, EP_LEN, actor, critic, model_actor_path="PPO2_model_actor.pth",
             model_critic_path="PPO2_model_critic.pth"):
    print('PPO2 Testing...')
    # Load pretrained actor and critic models
    checkpoint_actor = torch.load(model_actor_path)
    checkpoint_critic = torch.load(model_critic_path)
    actor.old_pi.load_state_dict(checkpoint_actor['net'])
    critic.critic_v.load_state_dict(checkpoint_critic['net'])
    total_rewards = []
    total_steps = []
    num_episode = 1

    for j in range(num_episode):
        actions = []
        state = env.reset()
        rewards = 0
        env._Render()
        for timestep in range(EP_LEN):
            obs_map, pos_map, goal_map = env._observe()
            obs_map = torch.tensor(obs_map, dtype=torch.float32)
            pos_map = torch.tensor(pos_map, dtype=torch.float32)
            goal_map = torch.tensor(goal_map, dtype=torch.float32)
            input_all_state = torch.stack((obs_map, pos_map, goal_map), dim=0).unsqueeze(0)

            # Take action and obtain reward
            action, action_logprob = actor.choose_action(input_all_state)
            actions.append(action[0])
            new_state, reward, done, info = env.step(action)
            time.sleep(0.5)  # Perform an action every 0.5 seconds
            env._Render()  # Render new state
            rewards += reward
            state = new_state

            if done:
                break

        print("steps: ", len(actions))
        print("reward:", rewards)

        total_rewards.append(rewards)
        total_steps.append(len(actions))

    '''
    average_reward = sum(total_rewards) / num_episode
    average_steps = sum(total_steps) / num_episode
    print("average_steps: ", average_steps)
    print("average_reward: ", average_reward)
    '''
    env.close()


if __name__ == "__main__":
    # Initialize actor and critic objects
    actor = PPO2Actor()
    critic = PPO2Critic()

    model_actor_path = "PPO2_Models/PPO2_model_actor.pth"
    model_critic_path = "PPO2_Models/PPO2_model_critic.pth"

    # Train or test based on Switch value
    if Switch == 0:
        ppo2train(env, EP_MAX, EP_LEN, BATCH, GAMMA, actor, critic, RENDER=False)
    else:
        ppo2test(env, EP_LEN, actor, critic, model_actor_path=model_actor_path,
                 model_critic_path=model_critic_path)
