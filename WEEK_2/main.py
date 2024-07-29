import argparse
import gymnasium as gym
import torch
from itertools import count
from tqdm import tqdm
from assets import DQNAgent, device

def main(args):
    env = gym.make('LunarLander-v2', render_mode='None')
    state_size = env.observation_space.shape[0]
    
    if isinstance(env.action_space, gym.spaces.Discrete):
        action_size = env.action_space.n
    elif isinstance(env.action_space, gym.spaces.Box):
        action_size = env.action_space.shape[0]
    else:
        raise ValueError("Unsupported action space type")

    agent = DQNAgent(state_size, action_size, args.eps_start, args.eps_end, args.eps_decay, args.gamma, args.lr, args.batch_size, args.tau)

    for i_episode in (range(args.episodes)):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        total_reward = 0
        for t in count():
            action = agent.select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            total_reward += reward

            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if done and terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            agent.memory.push(state, action, next_state, reward)

            state = next_state

            agent.optimize_model()

            target_net_state_dict = agent.target_net.state_dict()
            policy_net_state_dict = agent.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * args.tau + target_net_state_dict[key] * (1 - args.tau)
            agent.target_net.load_state_dict(target_net_state_dict)

            if done:
                if terminated and total_reward >= 200:  # 성공적인 착지로 간주
                    print(f"Episode {i_episode}: Successful landing! Total reward: {total_reward}")
                else:
                    print(f"Episode {i_episode}: Crash or failure. Total reward: {total_reward}")
                agent.episode_rewards.append(total_reward)
                agent.plot_rewards()
                break

        if i_episode % args.target_update == 0:
            agent.update_target_net()

    print('Complete')

    # 모델 저장
    agent.policy_net.to('cpu')
    print('now save!')
    torch.save(agent.policy_net.state_dict(), args.save_path)
    agent.policy_net.to(device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes to train the agent')
    parser.add_argument('--eps_start', type=float, default=0.9, help='Starting value of epsilon for epsilon-greedy policy')
    parser.add_argument('--eps_end', type=float, default=0.05, help='Ending value of epsilon for epsilon-greedy policy')
    parser.add_argument('--eps_decay', type=int, default=200, help='Epsilon decay factor')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--tau', type=float, default=0.005, help='Soft update coefficient for target network')
    parser.add_argument('--target_update', type=int, default=10, help='Number of episodes between target network updates')
    parser.add_argument('--save_path', type=str, default='dqn_lunarlander.pth', help='Path to save the trained model')
    args = parser.parse_args()
    main(args)
