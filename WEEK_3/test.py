import argparse
import gymnasium as gym
import torch
import json
from assets import PPO, device

def get_activation_fn(name):
    if name == "tanh":
        return torch.tanh
    elif name == "relu":
        return torch.relu
    elif name == "sigmoid":
        return torch.sigmoid
    else:
        raise ValueError(f"Unknown activation function: {name}")

def test(args):
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

    agent.policy.load_state_dict(torch.load(args.model_path))
    agent.policy.eval()

    episodes = 10
    try:
        total_total_reward = 0
        for i_episode in range(episodes):
            state, info = env.reset(seed=args.seed)
            total_reward = 0
            for t in range(args.max_timesteps):
                action, _ = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward


                if terminated or truncated:
                    break
                state = next_state

            # print(f"Episode {i_episode}, Total Reward: {total_reward}")
            total_total_reward += total_reward
        print(f"Average Total Reward: {total_total_reward / episodes}")
    finally:
        env.close()  # Ensure the environment is properly closed

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_timesteps", type=int, default=1500, help="Maximum timesteps per episode")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--model_path", type=str, default="ppo_hopper.pth", help="Path to the saved model")
    args = parser.parse_args()
    test(args)
