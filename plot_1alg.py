import os
import re
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from matplotlib.ticker import FuncFormatter


def extract_game_name(folder_name):
    match = re.match(r'ALE-(.+)-v5_\d+-\d+-\d+-\d+-\d+', folder_name)
    if match:
        return match.group(1)
    return None


from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def load_tensorboard_data(log_dir):
    # Load the TensorBoard data
    accumulator = EventAccumulator(log_dir)
    accumulator.Reload()

    # Check if the required scalars exist before extracting the data
    if 'avg_reward' not in accumulator.Tags()['scalars']:
        raise KeyError("Scalar 'avg_reward' not found in the log data.")
    if 'q' not in accumulator.Tags()['scalars']:
        raise KeyError("Scalar 'q' not found in the log data.")

    # Extract reward and Q-value data
    rewards = [scalar.value for scalar in accumulator.Scalars('avg_reward')]
    q_values = [scalar.value for scalar in accumulator.Scalars('q')]

    return rewards, q_values



def formatter(x, pos):
    return f'{int(x / 10)}'  # Assumes x is the index, every index is 100000 updates


def plot_each_data(game_data, exp_name):
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'serif',
        # 'text.usetex': True,
    }) # 设置字体大小适合学术论文
    plt.style.use('seaborn-v0_8-whitegrid')  # 使用seaborn的白色网格风格

    for game_name, data in game_data.items():
        rewards, q_values = zip(*data)

        reward_means = np.mean(rewards, axis=0)
        reward_medians = np.median(rewards, axis=0)
        reward_max = np.max(rewards, axis=0)
        reward_min = np.min(rewards, axis=0)

        q_means = np.mean(q_values, axis=0)
        q_medians = np.median(q_values, axis=0)
        q_max = np.max(q_values, axis=0)
        q_min = np.min(q_values, axis=0)

        fig, axs = plt.subplots(2, 1, figsize=(10, 10), dpi=300)  # 使用更高的DPI确保图像质量

        # 绘制奖励统计
        ax1 = axs[0]
        ax1.plot(reward_means, label='Mean Reward', color='navy')
        ax1.plot(reward_medians, linestyle='dashed', label='Median Reward', color='darkorange')
        ax1.fill_between(range(len(reward_means)), reward_min, reward_max, color='lightgray', alpha=0.5)
        ax1.set_title(f'Reward for {game_name}')
        ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{int(x * 1)}'))
        ax1.set_xlabel('Millions of updates')
        ax1.set_ylabel('Reward')
        ax1.legend(frameon=False)
        ax1.set_facecolor('#f0f0f0')
        ax1.grid(True, linestyle='--', linewidth=0.6, color='white', alpha=1)

        # 绘制Q值统计
        ax2 = axs[1]
        ax2.plot(q_means, label='Mean Q', color='darkgreen')
        ax2.plot(q_medians, linestyle='dashed', label='Median Q', color='crimson')
        ax2.fill_between(range(len(q_means)), q_min, q_max, color='lightgray', alpha=0.5)
        ax2.set_title(f'Q Values for {game_name}')
        ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{int(x * 0.1)}'))
        ax2.set_xlabel('Millions of updates')
        ax2.set_ylabel('Q value')
        ax2.legend(frameon=False)
        ax2.set_facecolor('#f0f0f0')
        ax2.grid(True, linestyle='--', linewidth=0.6, color='white', alpha=1)

        plt.tight_layout()
        if os.path.exists(f'./figures/'+exp_name+'/') is False:
            os.makedirs(f'./figures/'+exp_name+'/')
        plt.savefig(f'./figures/'+exp_name+f'/{game_name}_stats.png', bbox_inches='tight')  # 保存为PNG格式，确保边界正确显示


def plot_data3x2(game_data, exp_name):
    # 更新matplotlib配置以适应学术论文的要求
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'serif',
        'text.usetex': True,
    })
    plt.style.use('classic')

    # 准备数据处理
    def prepare_data(data):
        rewards, q_values = zip(*data)
        min_length = min(len(q) for q in q_values)
        q_values = [q[:min_length] for q in q_values]

        reward_means = np.mean(rewards, axis=0)
        reward_medians = np.median(rewards, axis=0)
        reward_max = np.max(rewards, axis=0)
        reward_min = np.min(rewards, axis=0)

        q_means = np.mean(q_values, axis=0)
        q_medians = np.median(q_values, axis=0)
        q_max = np.max(q_values, axis=0)
        q_min = np.min(q_values, axis=0)

        return reward_means, reward_medians, reward_max, reward_min, q_means, q_medians, q_max, q_min

    # 绘制奖励统计图
    fig, axs = plt.subplots(3, 2, figsize=(12, 20), dpi=300)
    axs = axs.ravel()
    for idx, (game_name, data) in enumerate(game_data.items()):
        reward_means, reward_medians, reward_max, reward_min, _, _, _, _ = prepare_data(data)

        ax = axs[idx]
        ax.plot(reward_means, label='Mean Reward', color='navy')
        ax.plot(reward_medians, linestyle='dashed', label='Median Reward', color='darkorange')
        ax.fill_between(range(len(reward_means)), reward_min, reward_max, color='lightgray', alpha=0.5)
        ax.set_title(f'{game_name}')
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{int(x)}'))
        ax.set_xlabel('Millions of updates')
        ax.set_ylabel('Reward')
        ax.legend(frameon=False, loc='upper left')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_facecolor('#f0f0f0')  # 设置子图的背景颜色
        ax.grid(True, linestyle='--', linewidth=0.6, color='white', alpha=1)  # 添加透明的网格线
    plt.tight_layout()
    # plt.savefig('./figures/dqn/tpami_rewards.eps', format='eps', bbox_inches='tight')
    plt.savefig('./figures/'+exp_name+' rewards.png', bbox_inches='tight')

    # 绘制Q值统计图
    fig, axs = plt.subplots(3, 2, figsize=(12, 20), dpi=300)
    axs = axs.ravel()
    for idx, (game_name, data) in enumerate(game_data.items()):
        _, _, _, _, q_means, q_medians, q_max, q_min = prepare_data(data)

        ax = axs[idx]
        ax.plot(q_means, label='Mean Q', color='darkgreen')
        ax.plot(q_medians, linestyle='dashed', label='Median Q', color='crimson')
        ax.fill_between(range(len(q_means)), q_min, q_max, color='lightgray', alpha=0.5)
        ax.set_title(f'{game_name}')
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{int(x  )}'))
        ax.set_xlabel('Millions of updates')
        ax.set_ylabel('Q value')
        ax.legend(frameon=False, loc='upper left')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_facecolor('#f0f0f0')  # 设置子图的背景颜色
        ax.grid(True, linestyle='--', linewidth=0.6, color='white', alpha=1)  # 添加透明的网格线
    plt.tight_layout()
    # plt.savefig('./figures/dqn/tpami_q_values.eps', format='eps', bbox_inches='tight')
    plt.savefig('./figures/'+exp_name+' q values.png', bbox_inches='tight')


# 主执行函数
def main(log_dir, exp_name):
    game_data = {}
    for folder in os.listdir(log_dir):
        if '.log' in folder:
            continue
        game_name = extract_game_name(folder)
        if game_name:
            data_path = os.path.join(log_dir, folder)
            rewards, q_values = load_tensorboard_data(data_path)
            if game_name not in game_data:
                game_data[game_name] = []
            game_data[game_name].append((rewards, q_values))
    plot_each_data(game_data, exp_name)
    plot_data3x2(game_data, exp_name)


# 示例调用
main('./exps/double_dqn', exp_name='Double DQN')
