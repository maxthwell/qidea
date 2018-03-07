#coding=utf-8
import random
class CARD():
	def __init__(self,_type,_size):
		assert (_type in range(4))
		assert (_size in range(13))
		self._type=_type
		self._size=_size

def PLAYER():
	def __init__(self,id):
		assert(id in range(6))
		self.id=id
		self.cards=[]

class GAME():
	def __init__(self,nPlayers,nMaxLoop=10):
		self.nPlayers=nPlayers
		self.players=[PLAYER(id) for id in range(nPlayers)]
		self.out_players=[]
		self.cards=[]
		for i in range(4):
			for j in range(13):
				cards.append(CARD(i,j))
		random.shuffle(cards)
		self.action_history=[]
		self.actions=[
		'none',
		'see_cards',
		'give_up',
		'add_money_1',
		'add_money_2',
		'add_money_4',
		'add_money_8',
		]
		
	
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
		

	def play(self):
		self.deal()
		for i in range(nMaxLoop):
			for p in range(self.players)
				a=p.select_action():
				if a != 1:
					p.push_reward(-1)
					self.out(p)
				self.action_history.append([p.id,a])
			for player in range(self.player)
				a=player.select_action():
				if a==0 or a==1:
					player.push_reword(-1)
					self.out(p)
				
			
				
		
