import numpy as np
import random
from players import connect4Player
from connect4 import connect4
from copy import deepcopy

class monteCarloAI(connect4Player):
	'''
	For each legal first_move, monteCarloAI will simulate many random games
	starting from that legal move where each player plays random moves until the game is over. 
	monteCarloAI will keep track of which first_move lead to the most wins and play that move
	'''

	def play(self, env: connect4, move_dict: dict) -> None:

		random.seed(self.seed)

		env = deepcopy(env)
		env.visualize = False

		# Find legal moves
		# Determine which columns have an empty space and are thus playable
		possible = env.topPosition >= 0
		indices = []
		for i, p in enumerate(possible):
			if p: indices.append(i)

		# Init fitness trackers to track which first_move lead to the most wins
		vs = np.zeros(7)

		counter = 0

		# Number of similations to try and run before reaching time limit 
		num_sims = 1001

		save_increment = 50
	
		# Simulate 
		while counter < num_sims + 1:

			# Pick a random first_move
			first_move = random.choice(indices)

			# Play a random game until the game ends
			turnout = self.playRandomGame(deepcopy(env), first_move)

			# Track who won the random game
			if turnout == self.position:
				vs[first_move] += 1
			elif turnout != 0:
				vs[first_move] -= 1

			# Every save_increment games, record the best move so far 
			# (in case the time limit gets reach before while loop exits)
			if counter % save_increment == 0:

				# Best move is the first_move that accumulated the most random wins
				move_dict['move'] = np.argmax(vs)
			
			counter += 1
		
		move_dict['move'] = np.argmax(vs)

	def playRandomGame(self, env, first_move: int):
		''' 
		Play a game from the current game state of env where each player 
		plays random moves until the game it over
		Return which player won the game
		'''
		switch = {1:2,2:1}
		move = first_move
		player = self.position
		self.simulateMove(env, move, player)

		# Play until game is over
		while not env.gameOver(move, player):
			player = switch[player] # switch which player is playing

			# Calculate possible moves
			possible = env.topPosition >= 0 
			indices = []
			for i, p in enumerate(possible):
				if p: indices.append(i)
			
			# Select random legal move
			move = random.choice(indices)

			# Play move
			self.simulateMove(env, move, player)

		# If the game has a winner return the last player to play
		if env.is_winner: 
			return player
		
		# Else the game is a tie
		else: 
			return 0

	def simulateMove(self, env: connect4, move: int, player: int):
		'''
		Play the move
		'''
		env.board[env.topPosition[move]][move] = player
		env.topPosition[move] -= 1
		env.history[0].append(move)

