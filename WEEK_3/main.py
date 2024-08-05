import argparse
import gymnasium as gym
import torch
import json
from assets import PPO as PPO, device
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def get_activation_fn(name):
    if name == "tanh":
        return torch.tanh
    elif name == "relu":
        return torch.relu
    elif name == "sigmoid":
        return torch.sigmoid
    else:
        raise ValueError(f"Unknown activation function: {name}")

def main(args):
    with open("config.json") as f:
        config = json.load(f)

    activation_fn = get_activation_fn(config["activation_fn"])
    
    env = gym.make("Hopper-v4")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = PPO(
        state_dim,
        action_dim,
        hidden_dims=config['hidden_dims'],
        activation_fn=activation_fn,
        n_steps=config["n_steps"],
        n_epochs=config["n_epochs"],
        batch_size=config["batch_size"],
        policy_lr=config["policy_lr"],
        value_lr=config["value_lr"],
        gamma=config["gamma"],
        lmda=config["lmda"],
        clip_ratio=config["clip_ratio"],
        vf_coef=config["vf_coef"],
        ent_coef=config["ent_coef"],
    )
    episodes = 500

    rewards = []
    avg_rewards = []
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot(rewards, label="Total Reward")
    avg_line, = ax.plot(avg_rewards, label="10-Episode Avg Reward")
    ax.set_xlim(0, episodes)
    ax.set_ylim(-100, 700)
    ax.legend()

    for i_episode in tqdm(range(episodes)):
        state, info = env.reset(seed=args.seed)
        total_reward = 0
        for t in range(args.max_timesteps):
            action, log_prob = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            transition = (state, action, reward, next_state, terminated)
            
            agent.step(transition)

            if terminated or truncated:
                break
            state = next_state

        rewards.append(total_reward)
        if len(rewards) >= 10:
            avg_rewards.append(np.mean(rewards[-10:]))
        else:
            avg_rewards.append(np.mean(rewards))

        line.set_ydata(rewards)
        line.set_xdata(range(len(rewards)))
        avg_line.set_ydata(avg_rewards)
        avg_line.set_xdata(range(len(avg_rewards)))
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.01)
        print(f"Episode {i_episode}, Total Reward: {total_reward}")

    plt.ioff()
    plt.show()

    torch.save(agent.policy.state_dict(), args.save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_timesteps", type=int, default=1500, help="Maximum timesteps per episode")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--save_path", type=str, default="ppo_hopper.pth", help="Path to save the trained model")
    args = parser.parse_args()
    main(args)
