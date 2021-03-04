from agent.agent import *
import os
import sys
sys.path.append('./')  # 此时假设此 py 文件和 env 文件夹在同一目录下
import env

parser = argparse.ArgumentParser(description='Train or test neural net motor controller.')
parser.add_argument('--train', dest='train', action='store_true', default=True)
parser.add_argument('--test', dest='test', action='store_false')
args = parser.parse_args()

if __name__ == '__main__':
    # 初始化环境
    env = gym.make(ENV_NAME)
    env = env.unwrapped

    # reproducible，设置随机种子，为了能够重现
    env.seed(RANDOMSEED)
    np.random.seed(RANDOMSEED)
    tf.random.set_seed(RANDOMSEED)

    # 定义状态空间，动作空间，动作幅度范围
    s_dim = 50
    a_dim = 50
    a_bound = env.action_space.high

    print('s_dim', s_dim)
    print('a_dim', a_dim)

    # 用DDPG算法
    ddpg = DDPG(a_dim, s_dim, a_bound)

    # 训练部分：
    if args.train:  # train

        reward_buffer = []  # 用于记录每个EP的reward，统计变化
        t0 = time.time()  # 统计时间
        for i in range(MAX_EPISODES):
            t1 = time.time()
            s = env.reset()
            ep_reward = 0  # 记录当前EP的reward
            for j in range(MAX_EP_STEPS):
                # Add exploration noise
                a = ddpg.choose_action(s)  # 这里很简单，直接用actor估算出a动作

                # 为了能保持开发，这里用了另外一种方式增加探索。
                # 因此需要需要以a为均值，VAR为标准差，建立正态分布，再从正态分布采样出a
                # 因为a是均值，所以a的概率是最大的。但a相对其他概率由多大，是靠VAR调整。这里我们其实可以增加更新VAR，动态调整a的确定性
                # 然后进行裁剪

                # Question: 原文的正态分布为 N(0, \sigma^2)，按参考论文意思应该是 a + N(0, \sigma^2)
                a = np.clip(np.random.normal(loc=a, scale=sigma), 0, 1)
                # 与环境进行互动
                s_, r, done, info = env.step(a)

                # 保存s，a，r，s_
                ddpg.store_transition(s, a, r, s_)

                # 第一次数据满了，就可以开始学习
                if ddpg.pointer > MEMORY_CAPACITY:
                    ddpg.learn()

                # 输出数据记录
                s = s_
                ep_reward += r  # 记录当前EP的总reward
                if j == MAX_EP_STEPS - 1:
                    print(
                        '\rEpisode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
                            i, MAX_EPISODES, ep_reward,
                            time.time() - t1
                        ), end=''
                    )
                # plt.show()
            # test
        #     if i and not i % TEST_PER_EPISODES:
        #         t1 = time.time()
        #         s = env.reset()
        #         ep_reward = 0
        #         for j in range(MAX_EP_STEPS):
        #
        #             a = ddpg.choose_action(s)  # 注意，在测试的时候，我们就不需要用正态分布了，直接一个a就可以了。
        #             s_, r, done, info = env.step(a)
        #
        #             s = s_
        #             ep_reward += r
        #             if j == MAX_EP_STEPS - 1:
        #                 print(
        #                     '\rEpisode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
        #                         i, MAX_EPISODES, ep_reward,
        #                         time.time() - t1
        #                     )
        #                 )
        #
        #                 reward_buffer.append(ep_reward)
        #
        #     if reward_buffer:
        #         plt.ion()
        #         plt.cla()
        #         plt.title('DDPG')
        #         plt.plot(np.array(range(len(reward_buffer))) * TEST_PER_EPISODES, reward_buffer)  # plot the episode vt
        #         plt.xlabel('episode steps')
        #         plt.ylabel('normalized state-action value')
        #         plt.ylim(-2000, 0)
        #         plt.show()
        #         plt.pause(0.1)
        # plt.ioff()
        # plt.show()
        print('\nRunning time: ', time.time() - t0)
        ddpg.save_ckpt()

    # test
    ddpg.load_ckpt()
    s = env.reset()
    for i in range(MAX_EP_STEPS):
        # env.render()
        s, r, done, info = env.step(ddpg.choose_action(s))
        if done:
            break
