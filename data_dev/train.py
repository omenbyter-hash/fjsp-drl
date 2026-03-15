import copy
import json
import os
import random
import time
from collections import deque

import gym
import pandas as pd
import torch
import numpy as np
from visdom import Visdom

import PPO_model
from env.case_generator import CaseGenerator
from validate import validate, get_validate_env

'''
1. 环境参数 (env_paras)
这部分定义了车间（Environment）的基本属性，也就是AI要解决的问题规模。
num_jobs: 工件数量。训练时使用的工件个数（例如 10 或 20）。
num_mas: 机器数量。车间内可用的机器总数（例如 5 或 10）。
batch_size: 并行环境数量。训练时同时运行多少个车间模拟器来收集数据。数值越大，收集数据越快，但显存占用越高。
valid_batch_size: 验证集批量大小。在验证阶段（考试）时同时测试的实例数量。

2. 模型参数 (model_paras)
这部分定义了神经网络（大脑）的结构和容量。
in_size_ma: 机器原始特征维度。输入到网络的机器节点特征向量长度（代码中通常包含机器当前利用率等信息）。
out_size_ma: 机器嵌入维度。经过图神经网络处理后，机器节点输出向量的长度。
in_size_ope: 工序原始特征维度。输入到网络的工序节点特征向量长度（如加工时间、是否完成等）。
out_size_ope: 工序嵌入维度。经过图神经网络处理后，工序节点输出向量的长度。
hidden_size_ope: 隐藏层维度。操作节点嵌入过程中MLP（多层感知机）的隐藏层大小。
n_latent_actor: Actor 网络隐藏层维度。策略网络（决定动作）内部隐藏层的大小。
n_latent_critic: Critic 网络隐藏层维度。价值网络（评估状态）内部隐藏层的大小。
n_hidden_actor: Actor 网络层数。策略网络的深度。
n_hidden_critic: Critic 网络层数。价值网络的深度。
action_dim: 动作维度。通常这里固定为 1，因为在 PPO 中输出的是动作的概率分布或索引，但具体取决于代码实现。
num_heads: 注意力头数。列表格式（如 [1, 1]），表示每层图注意力网络（GAT）中使用的注意力头数量。
dropout: 丢弃率。防止过拟合的正则化参数，0.0 表示不丢弃。

3. 训练参数 (train_paras)
这部分定义了 PPO 算法的学习规则，就像老师的教学大纲。
lr: 学习率 (Learning Rate)。控制模型参数更新的步长，太大会震荡，太小收敛慢。
betas: Adam 优化器参数。通常是 [0.9, 0.999]，控制梯度的一阶和二阶矩估计。
gamma: 折扣因子 (Discount Factor)。决定了AI有多看重未来的奖励。0.99 意味着AI很有远见，1.0 意味着每一步奖励同等重要。
eps_clip: PPO 裁剪阈值。PPO算法的核心参数（通常 0.2），限制策略更新的幅度，保证训练稳定。
K_epochs: 更新次数。每次收集完数据后，利用这批数据对网络进行多少次重复训练。
A_coeff: Actor 损失系数。策略损失在总损失中的权重。
vf_coeff: Value 损失系数。价值函数损失在总损失中的权重。
entropy_coeff: 熵正则化系数。鼓励AI多尝试不同动作（探索），防止过早陷入局部最优。
minibath_size: 小批量大小。在 PPO 更新时，每次从收集到的数据中抽取多少样本进行梯度计算。
update_timestep: 更新频率。每隔多少次迭代（Iteration）更新一次网络参数。
save_timestep: 保存频率。每隔多少次迭代进行一次验证并尝试保存最佳模型。
max_iterations: 最大迭代次数。整个训练过程总共跑多少轮。
parallel_iter: 换题频率。每隔多少轮重新生成一批新的随机环境实例。
viz: 可视化开关。true 或 false，是否使用 Visdom 实时画图。
viz_name: 可视化环境名。Visdom 中的窗口名称。
'''

def setup_seed(seed):
    # 下面这几行都是为了固定随机数种子
    torch.manual_seed(seed)  # 设置 PyTorch CPU 的种子
    torch.cuda.manual_seed_all(seed)  # 设置 PyTorch GPU 的种子
    np.random.seed(seed)  # 设置 Numpy 的种子
    random.seed(seed)  # 设置 Python 原生 random 的种子
    torch.backends.cudnn.deterministic = True  # 保证卷积操作也是确定性的

def main():
    # PyTorch 初始化
    # gpu_tracker = MemTracker()  # Used to monitor memory (of gpu) 用于监控GPU显存情况
    # 检测是否有显卡
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        # 如果有显卡，默认创建的张量（数据）都在显卡上
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    print("PyTorch device: ", device.type)
    torch.set_printoptions(precision=None, threshold=np.inf, edgeitems=None, linewidth=None, profile=None, sci_mode=False)

    # 读取同目录下的 config.json 文件
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'config.json')
    with open(config_path, 'r') as load_f:
        load_dict = json.load(load_f)

    # 把配置拆分成三份：环境参数、模型参数、训练参数
    env_paras = load_dict["env_paras"]
    model_paras = load_dict["model_paras"]
    train_paras = load_dict["train_paras"]

    # 把刚才检测到的 device (CPU/GPU) 塞进参数字典里，传给模型用
    env_paras["device"] = device
    model_paras["device"] = device
    env_valid_paras = copy.deepcopy(env_paras)
    env_valid_paras["batch_size"] = env_paras["valid_batch_size"]
    model_paras["actor_in_dim"] = model_paras["out_size_ma"] * 2 + model_paras["out_size_ope"] * 2
    model_paras["critic_in_dim"] = model_paras["out_size_ma"] + model_paras["out_size_ope"]

    num_jobs = env_paras["num_jobs"]
    num_mas = env_paras["num_mas"]
    opes_per_job_min = int(num_mas * 0.8)
    opes_per_job_max = int(num_mas * 1.2)

    # 创建一个记忆库，用来存放训练过程中产生的数据（状态、动作、奖励）
    memories = PPO_model.Memory()
    # 实例化 PPO 算法模型（你的智能体）
    model = PPO_model.PPO(model_paras, train_paras, num_envs=env_paras["batch_size"])
    # 创建验证环境（用来考试的题目，看看模型学得怎么样）
    env_valid = get_validate_env(env_valid_paras)

    # 用来保存最好的模型
    maxlen = 1  # Save the best model
    best_models = deque()
    makespan_best = float('inf')

    # 这里的 viz 是 Visdom 工具，用来在网页上实时画训练曲线
    # 如果你没有启动 visdom server，这行代码下面的 viz.line 会报错
    is_viz = train_paras["viz"]
    if is_viz:
        viz = Visdom(env=train_paras["viz_name"])

    # 创建保存训练结果的文件夹，名字带时间戳，防止覆盖
    str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
    save_path = './save/train_{0}'.format(str_time)
    os.makedirs(save_path)
    valid_results = []
    valid_results_100 = []

    # 初始化 Excel 文件（只写入列标题）
    data_file = pd.DataFrame(np.arange(10, 1010, 10), columns=["iterations"])
    data_file.to_excel('{0}/training_ave_{1}.xlsx'.format(save_path, str_time), sheet_name='Sheet1', index=False)
    data_file.to_excel('{0}/training_100_{1}.xlsx'.format(save_path, str_time), sheet_name='Sheet1', index=False)

    # 开始一轮轮的训练迭代 (Iteration)
    start_time = time.time()
    env = None
    for i in range(1, train_paras["max_iterations"] + 1):
        # 1. 生成新题目 (Instance Generation)
        # 每隔 parallel_iter 轮（比如20轮），就生成一批新的随机调度任务
        if (i - 1) % train_paras["parallel_iter"] == 0:
            # \mathcal{B} instances use consistent operations to speed up training
            nums_ope = [random.randint(opes_per_job_min, opes_per_job_max) for _ in range(num_jobs)]
            # 生成具体的 FJSP 案例
            case = CaseGenerator(num_jobs, num_mas, opes_per_job_min, opes_per_job_max, nums_ope=nums_ope)
            # 创建 Gym 环境，把生成的案例塞进去
            env = gym.make('fjsp-v0', case=case, env_paras=env_paras)
            env.reset()
            print('num_job: ', num_jobs, '\tnum_mas: ', num_mas, '\tnum_opes: ', sum(nums_ope))

        # Get state and completion signal
        state = env.unwrapped.state
        done = False
        dones = env.unwrapped.done_batch
        last_time = time.time()

        # 2. 玩游戏 (Rollout / Sampling)
        # 一直循环，直到所有工件都加工完 (done=True)
        while not done:
            with torch.no_grad():  # 玩游戏时不需要计算梯度，省显存
                # 让智能体根据当前状态(state)决定动作(actions)
                actions = model.policy_old.act(state, memories, dones)
            # 环境执行动作，返回：新状态、奖励、是否结束
            state, rewards, dones, _ = env.step(actions) #智能体尝试排班，直到排完。
            done = dones.all()
            # 把奖励和结束标记存进记忆库，留着后面学习用
            memories.rewards.append(rewards)
            memories.is_terminals.append(dones)
            # gpu_tracker.track()  # Used to monitor memory (of gpu)
        print("spend_time: ", time.time() - last_time)

        # 3. 验证一下刚才生成的调度方案是否合法（有没有撞机等bug）
        gantt_result = env.validate_gantt()[0]
        if not gantt_result:
            print("Scheduling Error！！！！！！")
        # print("Scheduling Finish")
        env.reset()

        # 4. 更新模型 (PPO Update) —— 真正变聪明的一步
        # 并不是每玩一局就更新，而是攒够了 update_timestep 局才更新一次
        if i % train_paras["update_timestep"] == 0:
            # 调用 PPO 的 update 函数，利用记忆库里的数据计算梯度，更新神经网络
            loss, reward = model.update(memories, env_paras, train_paras)
            # 打印当前轮次的奖励和损失
            print("reward: ", '%.3f' % reward, "; loss: ", '%.3f' % loss)
            # 清空记忆库，准备存下一批数据
            memories.clear_memory()
            if is_viz:
                viz.line(X=np.array([i]), Y=np.array([reward]),
                    win='window{}'.format(0), update='append', opts=dict(title='reward of envs'))
                viz.line(X=np.array([i]), Y=np.array([loss]),
                    win='window{}'.format(1), update='append', opts=dict(title='loss of envs'))  # deprecated

        # if iter mod x = 0 then validate the policy (x = 10 in paper)
        # 每隔 save_timestep 轮（比如10轮）进行一次验证
        if i % train_paras["save_timestep"] == 0:
            print('\nStart validating')
            # 调用 validate 函数，在之前准备好的 env_valid 上测试
            # vali_result 是平均完工时间
            vali_result, vali_result_100 = validate(env_valid_paras, env_valid, model.policy_old)
            valid_results.append(vali_result.item())
            valid_results_100.append(vali_result_100)

            # 如果这次考得比以前都好 (完工时间更短)
            # Save the best model
            if vali_result < makespan_best:
                makespan_best = vali_result # 把旧模型删了，保存这个新模型
                if len(best_models) == maxlen:
                    delete_file = best_models.popleft()
                    os.remove(delete_file)
                save_file = '{0}/save_best_{1}_{2}_{3}.pt'.format(save_path, num_jobs, num_mas, i)
                best_models.append(save_file)
                torch.save(model.policy.state_dict(), save_file) # torch.save 保存的是神经网络的参数字典 (state_dict)

            if is_viz:
                viz.line(
                    X=np.array([i]), Y=np.array([vali_result.item()]),
                    win='window{}'.format(2), update='append', opts=dict(title='makespan of valid'))

    # Save the data of training curve to files
    with pd.ExcelWriter('{0}/training_ave_{1}.xlsx'.format(save_path, str_time), mode='a', if_sheet_exists='overlay') as writer:
        data = pd.DataFrame(np.array(valid_results).transpose(), columns=["res"])
        data.to_excel(writer, sheet_name='Sheet1', index=False, startcol=1)

    with pd.ExcelWriter('{0}/training_100_{1}.xlsx'.format(save_path, str_time), mode='a', if_sheet_exists='overlay') as writer:
        column = [i_col for i_col in range(100)]
        data = pd.DataFrame(np.array(torch.stack(valid_results_100, dim=0).to('cpu')), columns=column)
        data.to_excel(writer, sheet_name='Sheet1', index=False, startcol=1)

    print("total_time: ", time.time() - start_time)

if __name__ == '__main__':
    main()