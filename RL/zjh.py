#coding=utf-8
import random
class CARD():
	def __init__(self,_type,_size):
		assert (_type in range(4))
		assert (_size in range(13))
		self._type=_type
		self._size=_size

class PLAYER():
	def __init__(self,id):
		assert(id in range(6))
		self.id=id
		self.see_card=False
		self.cards=[]
	def see_card():
		self.see_card=True

class GAME():
	def __init__(self,nPlayers,nMaxLoop=10):
		self.nPlayers=nPlayers
		self.players=[PLAYER(id) for id in range(nPlayers)]	#保存未淘汰的用户
		self.out_players=[]  #保存淘汰掉的用户
		self.cards=[]	     #保存扑克牌
		self.reward_pool=0   #当前奖池累积
		self.min_add_money=1 #最小加注数
		for i in range(4):
			for j in range(13):
				cards.append(CARD(i,j))
		random.shuffle(cards)
		self.action_history=[]
		self.actions=[
		'none',
		'see_cards',
		'give_up',
		'compare_0',
		'compare_1',
		'compare_2',
		'compare_3',
		'compare_4',
		'compare_5',
		'add_money_1',
		'add_money_2',
		'add_money_4',
		'add_money_8',
		]
		
	#发牌操作
	def deal(self):
		for i range(3):
			for player in range(self.players):
				card=self.cards.pop()
				player.append(card)
	#淘汰一个玩家
	def out(self,p):
		p.history
		self.player.remove(p)
		self.out_player.append(p)

	
	#加注操作,每执行一次加注操作用户的钱就暂时损失一笔钱
	def add_money(self,p,money):
		self.reward(p,-1*money)

	#对player进行奖励
	def reward(self,p,money):
		self.reward_pool-=money
		return p.id,money

	#选择看牌阶段
	def gambling_1(self,epoch):
		for p in range(self.players)
			a=p.select_action():
			if a>1: #不允许执行除看牌或无操作以外的操作
				self.reward(p,-10)
				self.out(p)
			elif a==1:
				if p.see_card:#不允许二次看牌
					self.reward(p,-10)
					self.out(p)
				else:
					p.see_card()
			self.action_history.append([p.id,a])

	#选择加注、比牌、弃牌阶段
	def gambling_2(self,epoch):
		for p in range(self.player)
			#玩家数量只剩下一个人，游戏结束
			if len(self.players == 1):
				return 'game_over'
			a=p.select_action():
			if a==0 or a==1:#必须要操作，并且不是看牌操作
				#违反规则给与惩罚，并且该玩家出局
				self.reward(p,-10)
				self.out(p)
			if a==2:#选择弃牌
				self.out(p)
			if a in range(3,9):#选择比牌操作，但不能选择比牌对象是自身
				compare_id=a-3
				if p.id==compare_id:
					#违反规则给与惩罚，并且该玩家出局
					self.push_reward(p,-10)
					self.out(p)
				self.compare(p.id,compare_id)
			if a in range(9,13):#选择加注操作
				self.add_money(p.id,2**(a-9))
			#加入到游戏的动作历史集中，作为游戏状态的一个属性。
			self.action_history.append([p.id,a])
	
	#比牌操作
	def compare(self,p1,p2):
		#比牌者必须拿出两倍的最小加注数放入奖池
		self.reward(p1,-1*self.min_add_money*2)
		if p1>p2:
			self.out(p2)
		else:
			self.out(p1):

	#摊牌操作，所有玩家都亮出底牌，比较大小后给与奖励后游戏结束
	def showhand(self):
		pass

	#游戏结束
	def game_over(self):
		#将奖池中累积的奖励统统奖励给胜者，游戏结束
		assert(len(self.players)==1)
		p=self.player.pop()
		self.reward(p,self.reward_pool)

	def play(self):
		#清空奖励池
		self.reward_pool=0
		#洗牌、发牌
		self.deal()
		#玩家开始轮流操作
		for i in range(nMaxLoop+1):
			#到达轮数上限
			if i==nMaxLoop:
				self.showhand()
				break
			#选择是否看牌操作
			self.gambling_1()
			#选择加注、比牌、弃牌
			if 'game_over' == self.gambling_2(i):
				self.game_over()
			
		
				
			
				
		
