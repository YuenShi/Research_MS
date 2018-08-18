# Author： 施源 Kris
# Create Time： 2018.8.15

import StockEnvironment
import DQN
import Helper

# 全局变量

# 奖励折扣率
GAMMA = 0.9

# 贪心度
EPSILON = 1.0

# 训练轮次
# EPOCHS = 30


class Runner:
    def __init__(self):

        # 初始化奖励折扣率
        self.gamma = GAMMA
        # 初始化贪心度
        self.epsilon = EPSILON
        
    def trainer(self, env, epochs):
        # 获取环境
        self.env = env
        # 初始化dqn_agent
        self.dqn_agent = DQN.DQN(env=self.env)
        # 初始化执行轮次
        self.epochs = epochs
        # 初始化每轮次的训练步数
        self.epoch_len = self.env.sample_size - self.env.n_timesteps

        # 开始执行epochs轮次的训练
        for epoch in range(self.epochs):
            # 初始化训练环境
            self.env.reset()
            # 获取当前状态
            cur_state = self.env.states   #.reshape(-1)
            # 初始化资本总值记录
            fortune = list()
            # 初始化动作记录
            act = list()
            # 初始化奖励记录
            re = list()
            # 开始本轮的训练
            for step in range(self.epoch_len):
                # 根据当前状态选择action
                action = self.dqn_agent.act(cur_state)
                # 执行action，返回环境中的下一个状态
                new_state, done, reward = self.env.execute(action)

                # 记录当前采取的行动，添加到action记录表
                act.append(action)
                # 记录当前返回的奖励，添加到reward记录表
                re.append(reward)
                # 计算当前状态的资本总值，添加到fortune记录表
                _fortune = new_state[-1,6] + new_state[-1,7]
                fortune.append(_fortune)

                # 存入回放记忆
                self.dqn_agent.remember(cur_state, action, reward, new_state, done)
                # 经验回放
                self.dqn_agent.replay() 
                if step >= 20:
                    # 更新模型参数
                    self.dqn_agent.target_train() 
                    if step % 20 == 0:
                        self.dqn_agent.save_model("train-model-{}.h5".format(epoch))
                # 获取下一个状态
                cur_state = new_state

                # 若为结束状态，跳出循环；否则继续训练
                if done:
                    break
                
            # 画出本轮次的资本总值变化图
            print("Epoch " + str(epoch) + " : ")
            # Helper.plot_fortune(fortune)
            # 输出本轮次所有行为
            print("Actions for epoch " + str(epoch) + ":")
            print(act)
            # 输出本轮次所有奖励
            print("Rewards for epoch "+ str(epoch) + ":")
            print(re)
        
            # 若当前轮次的资本总值小于8万，存入失败模型
            if fortune[-1] <= 80000.:
                print("Failed to complete in trial {}".format(epoch))
                if step % 50 == 0:
                    self.dqn_agent.save_model("fail-model-{}.h5".format(epoch))
            # 若当前轮次的资本总值大于11万，存入成功模型
            elif fortune[-1] > 110000.:
                print("Completed in {} trials".format(epoch))
                self.dqn_agent.save_model("success-model-{}.h5".format(epoch))
            # 若非以上两者，不保存模型
            else:
                continue

        # 完成所有轮次训练后，返回训练后的dqn_agent       
        return self.dqn_agent
                       

    def tester(self, env, agent, epochs):
        # 根据测试数据，初始化环境
        self.env = env
        # 初始化训练轮次 —— 测试时可能因为违反交易规则被提前终止，所以多测几轮
        self.epochs  = epochs
        # 初始化每轮次的训练步数
        self.epoch_len = self.env.sample_size - self.env.n_timesteps
        # 初始化dqn_agent
        self.dqn_agent = agent
        # 用于存放最终预测结果
        self.predict_act = 0

        # 开始执行epochs轮次的测试
        for epoch in range(self.epochs):
            # 初始化测试环境
            self.env.reset()
            # 获取当前状态
            cur_state = self.env.states   #.reshape(-1)
            # 初始化资本总值记录
            fortune = list()
            # 初始化动作记录
            act = list()
            # 初始化奖励记录
            re = list()
            # 开始本轮的测试
            for step in range(self.epoch_len):
                # 根据当前状态选择action
                action = self.dqn_agent.act(cur_state)
                # 执行action，返回环境中的下一个状态
                new_state, done, reward = self.env.execute(action)
            
                # 记录当前采取的行动，添加到action记录表
                act.append(action)
                # 记录当前返回的奖励，添加到reward记录表
                re.append(reward)
                # 计算当前状态的资本总值，添加到fortune记录表
                _fortune = new_state[-1,6] +  new_state[-1,7]
                fortune.append(_fortune)

                # 获取下一个状态
                cur_state = new_state

                # 若为结束状态，跳出循环；否则继续训练            
                if done:
                    break
                
            # 画出本轮次的资本总值变化图
            print("Epoch " + str(epoch) + " : ")
            # Helper.plot_fortune(fortune)
            # 输出本轮次所有行为
            print("Actions for epoch " + str(epoch) + ":")
            print(act)
            # 输出本轮次所有奖励
            print("Rewards for epoch "+ str(epoch) + ":")
            print(re)

            # 若未因为违反规则被提前终止，则输出当前的预测行为，终止测试
            if fortune[-1] > 100000.:
                self.predict_act = act[-1]
                print("Predict Cuurrent Action :")
                print(self.predict_act)
                # break

        return self.predict_act


                