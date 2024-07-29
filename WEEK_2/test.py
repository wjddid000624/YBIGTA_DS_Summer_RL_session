import argparse
import gymnasium as gym
import torch
from assets import DQN, device

def test(args):
    env = gym.make('LunarLander-v2', render_mode='human')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    policy_net = DQN(state_size, action_size).to(device)
    policy_net.load_state_dict(torch.load(args.model_path, map_location=device))

    total_rewards = []

    for i_episode in range(args.test_episodes):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        total_reward = 0
        done = False
        while not done:
            with torch.no_grad():
                action = policy_net(state).max(1).indices.view(1, 1)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            total_reward += reward

            if terminated or truncated:
                done = True
            else:
                state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        total_rewards.append(total_reward)
        print(f"Episode {i_episode}: Total reward: {total_reward}")

    avg_reward = sum(total_rewards[-10:]) / 10
    print(f'Average reward over last 10 episodes: {avg_reward}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--test_episodes', type=int, default=10, help='Number of episodes to test the agent')
    args = parser.parse_args()
    test(args)
