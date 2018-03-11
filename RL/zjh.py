#coding=utf-8
import random
class CARD():
	def __init__(self,_type,_size):
		assert (_type in range(4))
		assert (_size in range(13))
		self._type=_type
		self._size=_size
	def __str__(self):
		return '[type:%s,size:%s]'%(self._type,self._size)	
	def __cmp__(self,s):
		if self._size<s._size:
			return -1
		return 1 if self._size>s._size else 0

class PLAYER():
	def __init__(self,id):
		assert(id in range(6))
		self.id=id
		self.see_card=False
		self.cards=[]
		self.money=0
		self.alive=True
	def seecard(self):
		self.see_card=True
	#游戏结束后进行反省,修改策略参数，以期使自身获取更多的累积奖励
	def rethink(self):
		pass
	def choose_action(self):
		return random.randint(0,14)
	def handle_reward(self,money):
		self.money+=money
	def size(self):
		c=self.cards
		if c[0]._size==c[1]._size and c[1]._size==c[2]._size:
			return 60000+c[0]._size
		if c[0]._type==c[1]._type and c[1]._type==c[2]._type:
			if c[2]._size==c[1]._size+1 and c[1]._size==c[0]._size+1:
				return 50000+c[0]._size
			return 40000+c[0]._size+c[1]._size*13+c[2]._size*13*13
		if c[2]._size==c[1]._size+1 and c[1]._size==c[0]._size+1:
			return 30000+c[0]._size
		if c[0]._size==c[1]._size:
			return 20000+c[1]._size*100+c[2]._size
		if c[1]==c[2]:
			return 20000+c[1]._size*100+c[0]._size
		if c[0]._size==2 and c[1]._size==3 and c[2]._size==5:
			return 235
		return 10000+c[0]._size+c[1]._size*13+c[2]._size*13*13	
	
	def __cmp__(self,s):
		if self.size()<s.size():
			return -1
		return 1 if self.size()>s.size() else 0
		
	def output(self):
		print('playerid=>',self.id,'cards=>',[str(card) for card in self.cards],'money=>',self.money,'size=>',self.size(),'alive=>',self.alive)

class GAME():
	def __init__(self,nPlayers,max_epoch=10):
		self.nPlayers=nPlayers
		self.players=[PLAYER(id) for id in range(nPlayers)]	#保存未淘汰的用户
		self.cards=[]	     #保存扑克牌
		self.reward_pool=0   #当前奖池累积
		self.min_add_money=1 #最小加注数
		self.max_epoch=max_epoch
		for i in range(4):
			for j in range(13):
				self.cards.append(CARD(i,j))
		random.shuffle(self.cards)
		self.action_history=[] #动作的历史记录
		self.actions=[  #可选动作
		'no_see',
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
		'add_money_16',
		'add_money_32',
		]
		
	#发牌操作
	def deal(self):
		for i in range(3):
			for p in self.players:
				card=self.cards.pop()
				p.cards.append(card)
		for p in self.players:
			p.cards.sort()
	#淘汰一个玩家
	def out(self,p):
		p.alive=False
	
	def alive_players(self):
		ap=0
		for p in self.players:
			if p.alive:
				ap+=1
		return ap	
	#加注操作,每执行一次加注操作用户的钱就暂时损失一笔钱
	def add_money(self,p,money):
		self.reward(p,-1*money)

	#对player进行奖励
	def reward(self,p,money):
		self.reward_pool-=money
		p.handle_reward(money)
		return p.id,money

	#选择看牌阶段
	def gambling_1(self,epoch):
		for p in self.players:
			if not p.alive:
				continue
			a=p.choose_action()
			self.action_history.append([p.id,a])
			if a>1: #不允许执行除看牌或无操作以外的操作
				self.reward(p,-10)
				self.out(p)
			elif a==1:
				if p.see_card:#不允许二次看牌
					self.reward(p,-10)
					self.out(p)
				else:
					p.seecard()
			if self.alive_players()<=1:
				return 'over'

	#选择加注、比牌、弃牌阶段
	def gambling_2(self,epoch):
		for p in self.players:
			if not p.alive:
				continue
			a=p.choose_action()
			if a==0 or a==1:#必须要操作，并且不是看牌操作
				#违反规则给与惩罚，并且该玩家出局
				self.reward(p,-10)
				self.out(p)
			if a==2:#选择弃牌
				self.out(p)
			if a in range(3,9):#选择比牌操作，但不能选择比牌对象是自身或淘汰的玩家
				compare_id=a-3
				pCp=None
				for p1 in self.players:
					if p1.id==compare_id:
						pCp=p1
						break
				if p.id==compare_id or not pCp:
					#违反规则给与惩罚，并且该玩家出局
					self.reward(p,-10)
					self.out(p)
				else:
					self.compare(p,pCp)
			if a in range(9,15):#选择加注操作
				money=2**(a-9)
				if self.min_add_money<=money:
					self.min_add_money=money
					self.add_money(p,2**(a-9))
				else:
					self.reward(p,-10)
					self.out(p)
			#加入到游戏的动作历史集中，作为游戏状态的一个属性。
			self.action_history.append([p.id,a])
			if self.alive_players()<=1:
				return 'over'
	
	#比牌操作
	def compare(self,p1,p2):
		#比牌者必须拿出两倍的最小加注数放入奖池
		self.reward(p1,-1*self.min_add_money*2)
		if p1>p2:
			self.out(p2)
		else:
			self.out(p1)


	#摊牌操作，所有玩家都亮出底牌，比较大小后给与奖励后游戏结束
	def showhand(self):
		maxSize=0
		for p in self.players:
			if p.alive and maxSize <p.size():
				maxSize=p.size()
		for p in self.players:
			if p.alive and p.size()<maxSize:
				self.out(p)
		ap=self.alive_players()
		for p in self.players:
			if p.alive and p.size()==maxSize:
				self.reward(p,self.reward_pool/ap)
				p.rethink()

	def play(self):
		#清空奖励池
		self.reward_pool=0
		#洗牌、发牌、押底
		self.deal()
		for p in self.players:
			self.reward(p,-1)
		#玩家开始轮流操作
		for epoch in range(self.max_epoch+1):
			#到达轮数上限
			if epoch==self.max_epoch:
				return self.showhand()
			#选择是否看牌操作
			#选择加注、比牌、弃牌
			if 'over'==self.gambling_1(epoch) or 'over'==self.gambling_2(epoch):
				return self.showhand()
			
	def output(self):
		print('reward_pool:',self.reward_pool)
		print('players:')
		for p in self.players:
			p.output()
		print('actions:')
		for a in self.action_history:
			print(a)
			pid=a[0]
			aid=a[1]
			print('player:%s %s'%(pid,self.actions[aid]))
		print('')

if __name__=='__main__':
	for i in range(100):
		game=GAME(nPlayers=6)
		game.play()
		print('game %d'%i)
		game.output()	
