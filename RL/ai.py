#coding=utf-8
'''
这是一个用PPO算法玩一个平衡杆游戏。其动作空间是连续的，更新方法为回合更新，而且奖励信息也是可以实时获得的。
相较于这个游戏，扎金花更加简单，其动作空间只有15个离散的值，游戏的步数不到二十步，完全可以一直玩到结束，得到最终的累积回报，而不必使用衰减法进行估算。
'''


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym

#扎金花状态分类
'''
	c1)玩家个数N
	c2所有玩家的历史动作集。
	c3)当前玩家的手牌。
	c4)当前玩家曾经的比牌对象的手牌
	c5)被淘汰的玩家组成的标志向量
	c6)当前桌面上的玩家组成的标志向量
	c7)当前桌面上总共累积的注数
	c8)当前玩家的id
	c9)当前的轮数
	c10)剩余的轮数
	c11)当前允许押注的最小暗注（明注是暗注的两倍）
	c12)所有玩家的id与座次号。(构建一个N*N的矩阵，其每一行每一列都具有one-hot形式)
	综上当前玩家可见的所有信息可以抽象为：
		[20] =>标识当前轮数(以到达的轮数全部置1，为到达的置0)
		[6,3] =>标识当前的玩家，所有玩家的存活情况,所有玩家是否看牌
		[6,3,4)] =>标识每个玩家的三张手牌的花色（不对当前玩家可见则全部填充0，知道在相应标志位填1）
		[6,3,13)] =>标识每个玩家的三张手牌的点子（不对当前玩家可见则全部填充0，知道在相应标志位填1）	
		[6,20,2,15]=>标识每个玩家的20轮所采取的各种动作。（初始化全部为0）
		[6,20,2]   =>标识20轮每个动作后的奖池累计注数
		[6,20,2,6,3] =>标识每个动作执行后每个玩家的剩余注数、是否存活，是否看牌。
	因此构建的模型输入6个张量共将近10000个分量，
	输出1）为一个[15]的向量,标识选择每一种动作的概率。激活函数softmax。
	输出2）为一个[15]的向量，当前状态下的采取各个动作的优势Advantage，激活函数tanh
'''


EP_MAX = 1000
EP_LEN = 200
GAMMA = 0.9
A_LR = 0.0001
C_LR = 0.0002
BATCH = 32
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
S_DIM, A_DIM = 3, 1
METHOD = [
	dict(name='kl_pen', kl_target=0.01, lam=0.5),	# KL penalty
	dict(name='clip', epsilon=0.2),					# Clipped surrogate objective, find this is better
][1]		# choose the method for optimization


class PPO(object):

	def __init__(self):
		self.sess = tf.Session()
		self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')

		# critic
		with tf.variable_scope('critic'):
			l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
			self.v = tf.layers.dense(l1, 1)
			self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
			self.advantage = self.tfdc_r - self.v
			self.closs = tf.reduce_mean(tf.square(self.advantage))
			self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

		# actor
		pi, pi_params = self._build_anet('pi', trainable=True)
		oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
		with tf.variable_scope('sample_action'):
			self.sample_op = tf.squeeze(pi.sample(1), axis=0)		# choosing action
		with tf.variable_scope('update_oldpi'):
			self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

		self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
		self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
		with tf.variable_scope('loss'):
			with tf.variable_scope('surrogate'):
				# ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
				ratio = pi.prob(self.tfa) / oldpi.prob(self.tfa)
				surr = ratio * self.tfadv
				print('ratio==>',ratio)
				print('adv==>',self.tfadv)
				print('pi==>',pi)
				print('pi.prob(tfa)==>',pi.prob(self.tfa))
			if METHOD['name'] == 'kl_pen':
				self.tflam = tf.placeholder(tf.float32, None, 'lambda')
				kl = tf.distributions.kl_divergence(oldpi, pi)
				self.kl_mean = tf.reduce_mean(kl)
				self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))
			else:	# clipping method, find this is better
				self.aloss = -tf.reduce_mean(tf.minimum(
					surr,
					tf.clip_by_value(ratio, 1.-METHOD['epsilon'], 1.+METHOD['epsilon'])*self.tfadv))

		with tf.variable_scope('atrain'):
			self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)

		tf.summary.FileWriter("log/", self.sess.graph)

		self.sess.run(tf.global_variables_initializer())

	def update(self, s, a, r):
		self.sess.run(self.update_oldpi_op)
		adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
		# adv = (adv - adv.mean())/(adv.std()+1e-6)		# sometimes helpful

		# update actor
		if METHOD['name'] == 'kl_pen':
			for _ in range(A_UPDATE_STEPS):
				_, kl = self.sess.run(
					[self.atrain_op, self.kl_mean],
					{self.tfs: s, self.tfa: a, self.tfadv: adv, self.tflam: METHOD['lam']})
				if kl > 4*METHOD['kl_target']:	# this in in google's paper
					break
			if kl < METHOD['kl_target'] / 1.5:	# adaptive lambda, this is in OpenAI's paper
				METHOD['lam'] /= 2
			elif kl > METHOD['kl_target'] * 1.5:
				METHOD['lam'] *= 2
			METHOD['lam'] = np.clip(METHOD['lam'], 1e-4, 10)	# sometimes explode, this clipping is my solution
		else:	# clipping method, find this is better (OpenAI's paper)
			[self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(A_UPDATE_STEPS)]

		# update critic
		[self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(C_UPDATE_STEPS)]

	def _build_anet(self, name, trainable):
		with tf.variable_scope(name):
			l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu, trainable=trainable)
			mu = 2 * tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable=trainable)
			sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus, trainable=trainable)
			norm_dist = tf.contrib.distributions.Normal(loc=mu, scale=sigma)
		params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
		return norm_dist, params

	def choose_action(self, s):
		s = s[np.newaxis, :]
		a = self.sess.run(self.sample_op, {self.tfs: s})[0]
		return np.clip(a, -2, 2)

	def get_v(self, s):
		if s.ndim < 2: s = s[np.newaxis, :]
		return self.sess.run(self.v, {self.tfs: s})[0, 0]

env = gym.make('Pendulum-v0').unwrapped
ppo = PPO()
all_ep_r = []

for ep in range(EP_MAX):
	s = env.reset()
	buffer_s, buffer_a, buffer_r = [], [], []
	ep_r = 0
	for t in range(EP_LEN):	   # in one episode
		env.render()
		a = ppo.choose_action(s)
		s_, r, done, _ = env.step(a)
		buffer_s.append(s)
		buffer_a.append(a)
		buffer_r.append((r+8)/8)	# normalize reward, find to be useful
		s = s_
		ep_r += r

		# update ppo
		if (t+1) % BATCH == 0 or t == EP_LEN-1:
			v_s_ = ppo.get_v(s_)
			discounted_r = []
			for r in buffer_r[::-1]:
				v_s_ = r + GAMMA * v_s_
				discounted_r.append(v_s_)
			discounted_r.reverse()

			bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
			buffer_s, buffer_a, buffer_r = [], [], []
			ppo.update(bs, ba, br)
	if ep == 0: all_ep_r.append(ep_r)
	else: all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)
	print(
		'Ep: %i' % ep,
		"|Ep_r: %i" % ep_r,
		("|Lam: %.4f" % METHOD['lam']) if METHOD['name'] == 'kl_pen' else '',
	)

plt.plot(np.arange(len(all_ep_r)), all_ep_r)
plt.xlabel('Episode');plt.ylabel('Moving averaged episode reward');plt.show()
