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


def load_tensorboard_data(log_dir):
    accumulator = EventAccumulator(log_dir)
    accumulator.Reload()

    # 提取reward和q值的数据
    rewards = [scalar.value for scalar in accumulator.Scalars('avg_reward')]
    q_values = [scalar.value for scalar in accumulator.Scalars('q')]

    return rewards, q_values


def formatter(x, pos):
    return f'{int(x / 10)}'  # Assumes x is the index, every index is 100000 updates


def plot_each_data(game_data, alg_name0, bg_game_data, alg_name1):
    # 更新matplotlib配置以适应学术论文的要求
    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'serif',
        # 'text.usetex': True,
    })
    plt.style.use('seaborn-v0_8-whitegrid')

    for bg_game_name, bg_data in bg_game_data.items():
        bg_rewards, bg_q_values = zip(*bg_data)
        if bg_game_name not in game_data:
            continue
        rewards, q_values = zip(*game_data[bg_game_name])

        min_length = min(len(q) for q in bg_q_values)
        bg_q_values = [q[:min_length] for q in bg_q_values]

        # data for rewards
        reward_means = np.mean(rewards, axis=0)
        reward_medians = np.median(rewards, axis=0)
        reward_max = np.max(rewards, axis=0)
        reward_min = np.min(rewards, axis=0)

        # data for duel_rewards
        bg_reward_means = np.mean(bg_rewards, axis=0)
        bg_reward_medians = np.median(bg_rewards, axis=0)
        bg_reward_max = np.max(bg_rewards, axis=0)
        bg_reward_min = np.min(bg_rewards, axis=0)

        # data for q_values
        q_means = np.mean(q_values, axis=0)
        q_medians = np.median(q_values, axis=0)
        q_max = np.max(q_values, axis=0)
        q_min = np.min(q_values, axis=0)

        # data for duel_q_values
        bg_q_means = np.mean(bg_q_values, axis=0)
        bg_q_medians = np.median(bg_q_values, axis=0)
        bg_q_max = np.max(bg_q_values, axis=0)
        bg_q_min = np.min(bg_q_values, axis=0)

        fig, axs = plt.subplots(2, 1, figsize=(10, 10), dpi=300)  # 调整总图大小以适应详细信息

        # 绘制奖励统计
        ax1 = axs[0]
        # Dueling DQN数据
        ax1.plot(reward_means, label=alg_name0 + ' Mean Reward', color='deepskyblue')  # 更鲜艳的颜色
        ax1.plot(reward_medians, linestyle='dashed', label=alg_name0 + ' Median Reward', color='blue')

        # DQN数据
        ax1.plot(bg_reward_means, label=alg_name1 + ' Mean Reward', color='tomato')  # 使用与Dueling DQN对比度高的颜色
        ax1.plot(bg_reward_medians, linestyle='dashed', label=alg_name1 + ' Median Reward', color='darkred')

        # 填充区域
        ax1.fill_between(range(len(reward_means)), reward_min, reward_max, color='skyblue', alpha=0.2)
        ax1.fill_between(range(len(bg_reward_means)), bg_reward_min, bg_reward_max, color='salmon', alpha=0.2)

        ax1.set_title(f'Reward for {bg_game_name}')
        ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{int(x)}'))
        ax1.set_xlabel('Millions of updates')
        ax1.set_ylabel('Reward')
        ax1.legend(frameon=False)
        ax1.set_facecolor('#f0f0f0')
        ax1.grid(True, linestyle='--', linewidth=0.6, color='white', alpha=1)  # 网格线更加显眼

        # 绘制Q值统计
        ax2 = axs[1]
        # Dueling DQN数据
        ax2.plot(q_means, label=alg_name0 + ' Mean Q', color='darkcyan')
        ax2.plot(q_medians, linestyle='dashed', label=alg_name0 + ' Median Q', color='teal')

        # DQN数据
        ax2.plot(bg_q_means, label=alg_name1 + ' Mean Q', color='darkolivegreen')
        ax2.plot(bg_q_medians, linestyle='dashed', label=alg_name1 + ' Median Q', color='olive')

        # 填充区域
        ax2.fill_between(range(len(q_means)), q_min, q_max, color='paleturquoise', alpha=0.2)
        ax2.fill_between(range(len(bg_q_means)), bg_q_min, bg_q_max, color='lightgreen', alpha=0.2)

        ax2.set_title(f'Q Values for {bg_game_name}')
        ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{int(x )}'))
        ax2.set_xlabel('Millions of updates')
        ax2.set_ylabel('Q value')
        ax2.legend(frameon=False)
        ax2.set_facecolor('#f0f0f0')
        ax2.grid(True, linestyle='--', linewidth=0.6, color='white', alpha=1)

        plt.tight_layout()
        if os.path.exists(f'./figures/'+alg_name0 + ' vs ' + alg_name1) is False:
            os.makedirs(f'./figures/'+alg_name0 + ' vs ' + alg_name1)
        plt.savefig(f'./figures/'+alg_name0 + ' vs ' + alg_name1 + f'/{bg_game_name}_stats.png', bbox_inches='tight')  # 保存图表


def plot_data3x3(game_data, alg_name0, bg_game_data, alg_name1):
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

        q_means = np.mean(q_values, axis=0)[:2000:]
        q_medians = np.median(q_values, axis=0)[:2000:]
        q_max = np.max(q_values, axis=0)[:2000:]
        q_min = np.min(q_values, axis=0)[:2000:]

        return reward_means, reward_medians, reward_max, reward_min, q_means, q_medians, q_max, q_min

    # 绘制奖励统计图
    fig, axs = plt.subplots(3, 3, figsize=(25, 15), dpi=300)
    axs = axs.ravel()
    idx = 0
    for (game_name, bg_data) in bg_game_data.items():
        bg_reward_means, bg_reward_medians, bg_reward_max, bg_reward_min, _, _, _, _ = prepare_data(bg_data)
        if game_name not in game_data:
            continue
        reward_means, reward_medians, reward_max, reward_min, _, _, _, _ = prepare_data(game_data[game_name])
        ax = axs[idx]
        ax.plot(bg_reward_means, label=alg_name1 + ' Mean Reward', color='navy')
        ax.plot(bg_reward_medians, linestyle='dashed', label=alg_name1 + ' Median Reward', color='darkorange')
        ax.plot(reward_means, label=alg_name0 + ' Mean Reward', color='deepskyblue')
        ax.plot(reward_medians, linestyle='dashed', label=alg_name0 + ' Median Reward', color='blue')
        ax.fill_between(range(len(bg_reward_means)), bg_reward_min, bg_reward_max, color='lightgray', alpha=0.5)
        ax.set_title(f'{game_name}')
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{int(x )}'))
        ax.set_xlabel('Millions of updates')
        ax.set_ylabel('Reward')
        ax.legend(frameon=False, loc='upper left')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_facecolor('#f0f0f0')  # 设置子图的背景颜色
        ax.grid(True, linestyle='--', linewidth=0.6, color='white', alpha=1)  # 添加透明的网格线
        idx += 1
    # 处理空白子图
    for i in range(idx, len(axs)):
        axs[i].set_visible(False)  # 隐藏空白子图的轴
    plt.tight_layout()
    plt.savefig('./figures/' + alg_name0 + ' vs ' + alg_name1 + '_rewards.png', bbox_inches='tight')

    # 绘制Q值统计图
    fig, axs = plt.subplots(3, 3, figsize=(25, 15), dpi=300)
    axs = axs.ravel()
    idx = 0
    for (game_name, data) in bg_game_data.items():
        _, _, _, _, bg_q_means, bg_q_medians, bg_q_max, bg_q_min = prepare_data(data)
        if game_name not in game_data:
            continue
        _, _, _, _, q_means, q_medians, q_max, q_min = prepare_data(game_data[game_name])
        ax = axs[idx]
        ax.plot(bg_q_means, label=alg_name1 + ' Mean Q', color='darkgreen')
        ax.plot(bg_q_medians, linestyle='dashed', label=alg_name1 + ' Median Q', color='crimson')
        ax.plot(q_means, label=alg_name0 + ' Mean Q', color='darkblue')
        ax.plot(q_medians, linestyle='dashed', label=alg_name0 + ' Median Q', color='cornflowerblue')
        ax.fill_between(range(len(bg_q_means)), bg_q_min, bg_q_max, color='lightgray', alpha=0.5)
        ax.set_title(f'{game_name}')
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{int(x )}'))
        ax.set_xlabel('Millions of updates')
        ax.set_ylabel('Q value')
        ax.legend(frameon=False, loc='upper left')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_facecolor('#f0f0f0')  # 设置子图的背景颜色
        ax.grid(True, linestyle='--', linewidth=0.6, color='white', alpha=1)  # 添加透明的网格线
        idx += 1
    # 处理空白子图
    for i in range(idx, len(axs)):
        axs[i].set_visible(False)  # 隐藏空白子图的轴
    plt.tight_layout()
    # plt.savefig('./figures/dqn/tpami_q_values.eps', format='eps', bbox_inches='tight')
    plt.savefig('./figures/' + alg_name0 + ' vs ' + alg_name1 + '_q_values.png', bbox_inches='tight')


# 主执行函数
def main(log_dir, alg_name0, bg_log_dir, alg_name1):
    game_data = {}
    bg_data = {}

    for folder in os.listdir(log_dir):
        if '.log' in folder:
            continue
        game_name = extract_game_name(folder)
        if game_name:
            data_path = os.path.join(log_dir, folder)
            rewards, q_values = load_tensorboard_data(data_path)
            if len(rewards) >= 100:
                if game_name not in game_data:
                    game_data[game_name] = []
                game_data[game_name].append((rewards[:200], q_values[:2000]))

    for folder in os.listdir(bg_log_dir):
        if '.log' in folder:
            continue
        game_name = extract_game_name(folder)
        if game_name:
            data_path = os.path.join(bg_log_dir, folder)
            rewards, q_values = load_tensorboard_data(data_path)
            if len(rewards) >= 200:
                if game_name not in bg_data:
                    bg_data[game_name] = []
                bg_data[game_name].append((rewards[:200], q_values[:2000]))
    plot_each_data(game_data, alg_name0, bg_data, alg_name1)
    plot_data3x3(game_data, alg_name0, bg_data, alg_name1)


# 示例调用
main('./exps/dueling_dqn', 'Dueling DQN', './exps/dqn', 'DQN 2015')
