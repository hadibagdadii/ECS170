import random
import pygame
import math
from connect4 import connect4
import sys
import numpy as np
from copy import deepcopy
import time

class connect4Player(object):
	def __init__(self, position, seed=0, CVDMode=False):
		self.position = position
		self.opponent = None
		self.seed = seed
		random.seed(seed)
		if CVDMode:
			global P1COLOR
			global P2COLOR
			P1COLOR = (227, 60, 239)
			P2COLOR = (0, 255, 0)

	def play(self, env: connect4, move_dict: dict) -> None:
		move_dict["move"] = -1

class humanConsole(connect4Player):
	'''
	Human player where input is collected from the console
	'''
	def play(self, env: connect4, move_dict: dict) -> None:
		move_dict['move'] = int(input('Select next move: '))
		while True:
			if int(move_dict['move']) >= 0 and int(move_dict['move']) <= 6 and env.topPosition[int(move_dict['move'])] >= 0:
				break
			move_dict['move'] = int(input('Index invalid. Select next move: '))

class humanGUI(connect4Player):
	'''
	Human player where input is collected from the GUI
	'''

	def play(self, env: connect4, move_dict: dict) -> None:
		done = False
		while(not done):
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					sys.exit()

				if event.type == pygame.MOUSEMOTION:
					pygame.draw.rect(screen, BLACK, (0,0, width, SQUARESIZE))
					posx = event.pos[0]
					if self.position == 1:
						pygame.draw.circle(screen, P1COLOR, (posx, int(SQUARESIZE/2)), RADIUS)
					else: 
						pygame.draw.circle(screen, P2COLOR, (posx, int(SQUARESIZE/2)), RADIUS)
				pygame.display.update()

				if event.type == pygame.MOUSEBUTTONDOWN:
					posx = event.pos[0]
					col = int(math.floor(posx/SQUARESIZE))
					move_dict['move'] = col
					done = True

class randomAI(connect4Player):
	'''
	connect4Player that elects a random playable column as its move
	'''

	def play(self, env: connect4, move_dict: dict) -> None:
		possible = env.topPosition >= 0
		indices = []
		for i, p in enumerate(possible):
			if p: indices.append(i)
		move_dict['move'] = random.choice(indices)

class stupidAI(connect4Player):
	'''
	connect4Player that will play the same strategy every time
	Tries to fill specific columns in a specific order 
	'''
	def play(self, env: connect4, move_dict: dict) -> None:
		possible = env.topPosition >= 0
		indices = []
		for i, p in enumerate(possible):
			if p: indices.append(i)
		if 3 in indices:
			move_dict['move'] = 3
		elif 2 in indices:
			move_dict['move'] = 2
		elif 1 in indices:
			move_dict['move'] = 1
		elif 5 in indices:
			move_dict['move'] = 5
		elif 6 in indices:
			move_dict['move'] = 6
		else:
			move_dict['move'] = 0

class minimaxAI(connect4Player):
    '''
    Connect4Player that implements the minimax algorithm WITHOUT alpha-beta pruning
    Optimized for a 1-second time limit.
    '''
    
    def __init__(self, position, seed=0, CVDMode=False):
        super().__init__(position, seed, CVDMode)
        self.max_depth = 4  # Reduced max depth to fit within 1-second time limit
        
    def play(self, env: connect4, move_dict: dict) -> None:
        start_time = time.time()
        
        # Get legal moves
        possible = env.topPosition >= 0
        indices = []
        for i, p in enumerate(possible):
            if p: 
                indices.append(i)
        
        # No legal moves
        if not indices:
            return
        
        # Only one legal move
        if len(indices) == 1:
            move_dict['move'] = indices[0]
            return
            
        # Check for immediate winning move (very fast check)
        winning_move = self.find_immediate_win(env, self.position)
        if winning_move is not None:
            move_dict['move'] = winning_move
            return
            
        # Check if opponent has a winning move we need to block (very fast check)
        blocking_move = self.find_immediate_win(env, self.opponent.position)
        if blocking_move is not None:
            move_dict['move'] = blocking_move
            return
        
        # Clone the environment to avoid modifying the original
        env_copy = deepcopy(env)
        
        # Critical fix: Disable visualization in copy to prevent pygame errors
        env_copy.visualize = False
        
        # Default to center column if available, otherwise first legal move
        center_col = env.shape[1] // 2
        if env.topPosition[center_col] >= 0:
            move_dict['move'] = center_col  # Prefer center column as default
        else:
            move_dict['move'] = indices[0]
        
        # Start with a very shallow search and increase depth if time allows
        for current_depth in range(1, self.max_depth + 1):
            # If we're running out of time, use the move from the previous iteration
            if time.time() - start_time > 0.7:  # 0.7 second safety margin
                break
                
            best_score = -float('inf')
            best_move = move_dict['move']  # Default to previous best move
            
            # Search all possible moves at current depth
            for move in indices:
                # Check time again within the loop
                if time.time() - start_time > 0.8:  # Stricter time check
                    break
                    
                # Simulate the move
                row = env_copy.topPosition[move]
                env_copy.board[row][move] = self.position
                env_copy.topPosition[move] -= 1
                
                # Evaluate the move with minimax
                score = self.minimax(env_copy, move, current_depth - 1, False, start_time)
                
                # Undo the move
                env_copy.board[row][move] = 0
                env_copy.topPosition[move] += 1
                
                # Update best move if necessary
                if score > best_score:
                    best_score = score
                    best_move = move
            
            # Update the move with the best at current depth
            move_dict['move'] = best_move
    
    def minimax(self, env, last_move, depth, is_maximizing, start_time):
        """
        Minimax algorithm implementation
        
        Args:
            env: The current game state
            last_move: The last move played
            depth: How many more layers to search
            is_maximizing: Whether the current player is maximizing or minimizing
            start_time: The time when the play function was called
        
        Returns:
            The score for the current state
        """
        # Time check to abort deep searches if approaching time limit
        if time.time() - start_time > 0.85:
            # Return a reasonable score based on current state to avoid timeouts
            return 0 if is_maximizing else 0
        
        # Get the current player based on is_maximizing
        current_player = self.opponent.position if is_maximizing else self.position
        last_player = self.position if is_maximizing else self.opponent.position
        
        # Check if the game is over
        is_terminal = self.check_game_over(env, last_move, last_player)
        
        # If terminal state or max depth, return evaluation
        if is_terminal or depth == 0:
            return self.evaluate(env)
        
        # Get legal moves
        possible = env.topPosition >= 0
        indices = []
        for i, p in enumerate(possible):
            if p: 
                indices.append(i)
        
        if is_maximizing:
            max_eval = -float('inf')
            for move in indices:
                # Time check within the loop
                if time.time() - start_time > 0.85:
                    return max_eval
                    
                # Simulate the move
                row = env.topPosition[move]
                env.board[row][move] = self.position
                env.topPosition[move] -= 1
                
                # Recursively evaluate
                eval_score = self.minimax(env, move, depth - 1, False, start_time)
                
                # Undo the move
                env.board[row][move] = 0
                env.topPosition[move] += 1
                
                max_eval = max(max_eval, eval_score)
            return max_eval
        else:
            min_eval = float('inf')
            for move in indices:
                # Time check within the loop
                if time.time() - start_time > 0.85:
                    return min_eval
                    
                # Simulate the move
                row = env.topPosition[move]
                env.board[row][move] = self.opponent.position
                env.topPosition[move] -= 1
                
                # Recursively evaluate
                eval_score = self.minimax(env, move, depth - 1, True, start_time)
                
                # Undo the move
                env.board[row][move] = 0
                env.topPosition[move] += 1
                
                min_eval = min(min_eval, eval_score)
            return min_eval
    
    def check_game_over(self, env, last_move, player):
        """
        Check if the game is over without using pygame display functions
        
        Args:
            env: The game environment
            last_move: The last move made
            player: The player who made the last move
            
        Returns:
            Boolean indicating if the game is over
        """
        # Find the row of the last move
        i = env.topPosition[last_move] + 1
        j = last_move
        
        # Check horizontal
        count = 0
        for c in range(max(0, j-3), min(j+4, env.shape[1])):
            if env.board[i][c] == player:
                count += 1
            else:
                count = 0
            if count >= 4:
                return True
                
        # Check vertical
        count = 0
        for r in range(max(0, i-3), min(i+4, env.shape[0])):
            if env.board[r][j] == player:
                count += 1
            else:
                count = 0
            if count >= 4:
                return True
                
        # Check diagonal (positive slope)
        count = 0
        for d in range(-3, 4):
            r, c = i+d, j+d
            if 0 <= r < env.shape[0] and 0 <= c < env.shape[1]:
                if env.board[r][c] == player:
                    count += 1
                else:
                    count = 0
                if count >= 4:
                    return True
                    
        # Check diagonal (negative slope)
        count = 0
        for d in range(-3, 4):
            r, c = i-d, j+d
            if 0 <= r < env.shape[0] and 0 <= c < env.shape[1]:
                if env.board[r][c] == player:
                    count += 1
                else:
                    count = 0
                if count >= 4:
                    return True
        
        # Check if board is full (tie)
        return all(env.topPosition[col] < 0 for col in range(env.shape[1]))
    
    def evaluate(self, env):
        """
        Evaluate the current board state - simplified for speed
        
        Returns:
            A score representing how good the position is for the player
        """
        score = 0
        board = env.board
        rows, cols = board.shape
        
        pos_weight = [[3, 4, 5, 7, 5, 4, 3],
                        [4, 6, 8, 10, 8, 6, 4],
                        [5, 8, 11, 13, 11, 8, 5],
                        [7, 10, 13, 15, 13, 10, 7],
                        [5, 8, 11, 13, 11, 8, 5],
                        [4, 6, 8, 10, 8, 6, 4],
                        [3, 4, 5, 7, 5, 4, 3]]
        ## calculate based off of yours and the opponents
        ## counts 1s in the board
        p1_position = np.count_nonzero(board == self.position)
        p2_position = np.count_nonzero(board == self.opponent.position)

        score += np.sum(p1_position * pos_weight)
        score -= np.sum(p2_position * pos_weight)

        # Fast check for immediate wins
        for player in [self.position, self.opponent.position]:
            multiplier = 1 if player == self.position else -1
            
            # Check horizontal
            for r in range(rows):
                for c in range(cols - 3):
                    window = [board[r][c+i] for i in range(4)]
                    if window.count(player) == 4:
                        return 1000 * multiplier
            
            # Check vertical
            for r in range(rows - 3):
                for c in range(cols):
                    window = [board[r+i][c] for i in range(4)]
                    if window.count(player) == 4:
                        return 1000 * multiplier
            
            # Check positive diagonal
            for r in range(rows - 3):
                for c in range(cols - 3):
                    window = [board[r+i][c+i] for i in range(4)]
                    if window.count(player) == 4:
                        return 1000 * multiplier
            
            # Check negative diagonal
            for r in range(3, rows):
                for c in range(cols - 3):
                    window = [board[r-i][c+i] for i in range(4)]
                    if window.count(player) == 4:
                        return 1000 * multiplier
        
        # Check only key patterns for speed (instead of all windows)
        # Prioritize center column
        center_col = cols // 2
        center_array = [row[center_col] for row in board]
        center_count = center_array.count(self.position)
        score += center_count * 3
        
        # Check for threatening patterns (3 in a row with an empty spot)
        # This is a simplified evaluation that checks fewer windows but is much faster
        for r in range(rows):
            for c in range(cols - 3):
                window = [board[r][c+i] for i in range(4)]
                score += self.quick_evaluate_window(window)
        
        for r in range(rows - 3):
            for c in range(cols):
                window = [board[r+i][c] for i in range(4)]
                score += self.quick_evaluate_window(window)
        
        return score
    
    def quick_evaluate_window(self, window):
        """
        Simplified window evaluation for speed
        """
        player = self.position
        opponent = self.opponent.position
        
        # Only check the most critical patterns
        if window.count(player) == 3 and window.count(0) == 1:
            return 5  # Three in a row with an empty space
        elif window.count(opponent) == 3 and window.count(0) == 1:
            return -4  # Block opponent's three in a row
            
        return 0
        
    def find_immediate_win(self, env, player):
        """
        Check if there's an immediate winning move for the given player - optimized for speed
        
        Args:
            env: The game environment
            player: The player to check for
            
        Returns:
            The column index of the winning move, or None if none exists
        """
        # Get legal moves
        possible = env.topPosition >= 0
        indices = []
        for i, p in enumerate(possible):
            if p: 
                indices.append(i)
                
        # Try center column first as it often leads to more wins
        ordered_indices = sorted(indices, key=lambda x: abs(x - env.shape[1]//2))
                
        # Check each move to see if it results in a win
        for move in ordered_indices:
            row = env.topPosition[move]
            
            # Skip if column is full
            if row < 0:
                continue
                
            # Make the move
            env.board[row][move] = player
            
            # Simple win check (horizontal)
            count = 0
            for c in range(max(0, move-3), min(move+4, env.shape[1])):
                if env.board[row][c] == player:
                    count += 1
                else:
                    count = 0
                if count >= 4:
                    env.board[row][move] = 0  # Undo move
                    return move
                    
            # Simple win check (vertical)
            count = 0
            for r in range(max(0, row-3), min(row+4, env.shape[0])):
                if env.board[r][move] == player:
                    count += 1
                else:
                    count = 0
                if count >= 4:
                    env.board[row][move] = 0  # Undo move
                    return move
                    
            # Simple win check (diagonal - positive slope)
            for offset in range(4):
                if all(0 <= row+offset-i < env.shape[0] and 
                       0 <= move+offset-i < env.shape[1] and 
                       env.board[row+offset-i][move+offset-i] == player 
                       for i in range(4)):
                    env.board[row][move] = 0  # Undo move
                    return move
            
            # Simple win check (diagonal - negative slope)  
            for offset in range(4):
                if all(0 <= row-offset+i < env.shape[0] and 
                       0 <= move+offset-i < env.shape[1] and 
                       env.board[row-offset+i][move+offset-i] == player 
                       for i in range(4)):
                    env.board[row][move] = 0  # Undo move
                    return move
            
            # Undo the move
            env.board[row][move] = 0
                
        return None


class alphaBetaAI(connect4Player):
    '''
    Connect4Player that implements the minimax algorithm WITH alpha-beta pruning
    Optimized for high win rate against randomAI and monteCarloAI.
    '''
    
    def __init__(self, position, seed=0, CVDMode=False):
        super().__init__(position, seed, CVDMode)
        self.max_depth = 4  # Fixed max depth as required
        
        # Position weight matrix - rewards control of the center and positions with multiple win paths
        self.pos_weight = np.array([
            [3, 4, 5, 7, 5, 4, 3],
            [4, 6, 8, 10, 8, 6, 4],
            [5, 8, 11, 13, 11, 8, 5],
            [5, 8, 11, 13, 11, 8, 5],
            [4, 6, 8, 10, 8, 6, 4],
            [3, 4, 5, 7, 5, 4, 3]
        ])
        
        # Track game state for adaptive strategy
        self.move_count = 0
        
        # Initialize dynamic weight ratios based on game phase
        self.weight_ratios = {
            'early': {
                'win': 10000,
                'position': 1.2,
                'pattern': 1.0,
                'trap': 15
            },
            'mid': {
                'win': 10000,
                'position': 1.0,
                'pattern': 1.5,
                'trap': 20
            },
            'late': {
                'win': 10000,
                'position': 0.8,
                'pattern': 2.0,
                'trap': 30
            }
        }
    
    def play(self, env: connect4, move_dict: dict) -> None:
        start_time = time.time()
        self.move_count += 1
        
        # Get legal moves
        possible = env.topPosition >= 0
        indices = []
        for i, p in enumerate(possible):
            if p: 
                indices.append(i)
        
        # No legal moves
        if not indices:
            return
        
        # Only one legal move
        if len(indices) == 1:
            move_dict['move'] = indices[0]
            return
            
        # Check for immediate winning move (very fast check)
        winning_move = self.find_immediate_win(env, self.position)
        if winning_move is not None:
            move_dict['move'] = winning_move
            return
            
        # Check if opponent has a winning move we need to block (very fast check)
        blocking_move = self.find_immediate_win(env, self.opponent.position)
        if blocking_move is not None:
            move_dict['move'] = blocking_move
            return
        
        # Clone the environment to avoid modifying the original
        env_copy = deepcopy(env)
        
        # Critical fix: Disable visualization in copy to prevent pygame errors
        env_copy.visualize = False
        
        # Default to center column if available, otherwise first legal move
        center_col = env.shape[1] // 2
        if env.topPosition[center_col] >= 0:
            move_dict['move'] = center_col  # Prefer center column as default
        else:
            move_dict['move'] = indices[0]
        
        # Optimize move order: check center column first, then columns near center
        ordered_indices = sorted(indices, key=lambda x: abs(x - env.shape[1]//2))
        
        best_score = -float('inf')
        best_move = move_dict['move']  # Default to previous best move
        alpha = -float('inf')
        beta = float('inf')
        
        # Search all possible moves at current depth
        for move in ordered_indices:
            # Check time to avoid timeouts
            if time.time() - start_time > 2.7:  # Stricter time check
                break
                
            # Simulate the move
            row = env_copy.topPosition[move]
            if row < 0:  # Skip full columns
                continue
                
            env_copy.board[row][move] = self.position
            env_copy.topPosition[move] -= 1
            
            # Evaluate the move with minimax and alpha-beta pruning
            score = self.minimax(env_copy, move, self.max_depth - 1, False, alpha, beta, start_time)
            
            # Undo the move
            env_copy.board[row][move] = 0
            env_copy.topPosition[move] += 1
            
            # Update best move if necessary
            if score > best_score:
                best_score = score
                best_move = move
            
            # Update alpha
            alpha = max(alpha, best_score)
        
        # Update the move with the best at current depth
        move_dict['move'] = best_move
    
    def minimax(self, env, last_move, depth, is_maximizing, alpha, beta, start_time):
        """
        Minimax algorithm implementation with alpha-beta pruning
        
        Args:
            env: The current game state
            last_move: The last move played
            depth: How many more layers to search
            is_maximizing: Whether the current player is maximizing or minimizing
            alpha: The best value the maximizer can guarantee
            beta: The best value the minimizer can guarantee
            start_time: The time when the play function was called
        
        Returns:
            The score for the current state
        """
        # Time check to abort deep searches if approaching time limit
        if time.time() - start_time > 2.85:
            # Return a reasonable score based on current state to avoid timeouts
            return 0 if is_maximizing else 0
        
        # Get the current player based on is_maximizing
        current_player = self.opponent.position if is_maximizing else self.position
        last_player = self.position if is_maximizing else self.opponent.position
        
        # Check if the game is over
        is_terminal = self.check_game_over(env, last_move, last_player)
        
        # If terminal state or max depth, return evaluation
        if is_terminal or depth == 0:
            return self.evaluate(env)
        
        # Get legal moves
        possible = env.topPosition >= 0
        indices = []
        for i, p in enumerate(possible):
            if p: 
                indices.append(i)
                
        # Optimize move order: check center column first, then columns near center
        ordered_indices = sorted(indices, key=lambda x: abs(x - env.shape[1]//2))
        
        if is_maximizing:
            max_eval = -float('inf')
            for move in ordered_indices:
                # Time check within the loop
                if time.time() - start_time > 2.85:
                    return max_eval
                    
                # Simulate the move
                row = env.topPosition[move]
                if row < 0:  # Skip full columns
                    continue
                    
                env.board[row][move] = self.position
                env.topPosition[move] -= 1
                
                # Recursively evaluate
                eval_score = self.minimax(env, move, depth - 1, False, alpha, beta, start_time)
                
                # Undo the move
                env.board[row][move] = 0
                env.topPosition[move] += 1
                
                max_eval = max(max_eval, eval_score)
                
                # Alpha-beta pruning
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Beta cutoff
                
            return max_eval
        else:
            min_eval = float('inf')
            for move in ordered_indices:
                # Time check within the loop
                if time.time() - start_time > 2.85:
                    return min_eval
                    
                # Simulate the move
                row = env.topPosition[move]
                if row < 0:  # Skip full columns
                    continue
                    
                env.board[row][move] = self.opponent.position
                env.topPosition[move] -= 1
                
                # Recursively evaluate
                eval_score = self.minimax(env, move, depth - 1, True, alpha, beta, start_time)
                
                # Undo the move
                env.board[row][move] = 0
                env.topPosition[move] += 1
                
                min_eval = min(min_eval, eval_score)
                
                # Alpha-beta pruning
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha cutoff
                
            return min_eval
    
    def check_game_over(self, env, last_move, player):
        """
        Check if the game is over without using pygame display functions
        
        Args:
            env: The game environment
            last_move: The last move made
            player: The player who made the last move
            
        Returns:
            Boolean indicating if the game is over
        """
        # Find the row of the last move
        i = env.topPosition[last_move] + 1
        j = last_move
        
        # Check horizontal
        count = 0
        for c in range(max(0, j-3), min(j+4, env.shape[1])):
            if env.board[i][c] == player:
                count += 1
            else:
                count = 0
            if count >= 4:
                return True
                
        # Check vertical
        count = 0
        for r in range(max(0, i-3), min(i+4, env.shape[0])):
            if env.board[r][j] == player:
                count += 1
            else:
                count = 0
            if count >= 4:
                return True
                
        # Check diagonal (positive slope)
        count = 0
        for d in range(-3, 4):
            r, c = i+d, j+d
            if 0 <= r < env.shape[0] and 0 <= c < env.shape[1]:
                if env.board[r][c] == player:
                    count += 1
                else:
                    count = 0
                if count >= 4:
                    return True
                    
        # Check diagonal (negative slope)
        count = 0
        for d in range(-3, 4):
            r, c = i-d, j+d
            if 0 <= r < env.shape[0] and 0 <= c < env.shape[1]:
                if env.board[r][c] == player:
                    count += 1
                else:
                    count = 0
                if count >= 4:
                    return True
        
        # Check if board is full (tie)
        return all(env.topPosition[col] < 0 for col in range(env.shape[1]))
    
    def evaluate(self, env):
        """
        Enhanced evaluation function with dynamic weighting based on game phase
        
        Returns:
            A score representing how good the position is for the player
        """
        # Get current game phase weights
        weights = self.get_phase_weights()
        
        board = env.board
        rows, cols = board.shape
        
        # Fast check for immediate wins
        for player in [self.position, self.opponent.position]:
            multiplier = 1 if player == self.position else -1
            
            # Check horizontal
            for r in range(rows):
                for c in range(cols - 3):
                    window = [board[r][c+i] for i in range(4)]
                    if window.count(player) == 4:
                        return weights['win'] * multiplier  # Higher value for winning positions
            
            # Check vertical
            for r in range(rows - 3):
                for c in range(cols):
                    window = [board[r+i][c] for i in range(4)]
                    if window.count(player) == 4:
                        return weights['win'] * multiplier
            
            # Check positive diagonal
            for r in range(rows - 3):
                for c in range(cols - 3):
                    window = [board[r+i][c+i] for i in range(4)]
                    if window.count(player) == 4:
                        return weights['win'] * multiplier
            
            # Check negative diagonal
            for r in range(3, rows):
                for c in range(cols - 3):
                    window = [board[r-i][c+i] for i in range(4)]
                    if window.count(player) == 4:
                        return weights['win'] * multiplier
        
        # Position-based score
        position_score = 0
        
        # Calculate weighted positions
        our_weighted_sum = 0
        opponent_weighted_sum = 0
        
        for r in range(rows):
            for c in range(cols):
                if board[r][c] == self.position:
                    our_weighted_sum += self.pos_weight[r][c]
                elif board[r][c] == self.opponent.position:
                    opponent_weighted_sum += self.pos_weight[r][c]
        
        # Add weighted position score
        position_score = (our_weighted_sum - opponent_weighted_sum) * weights['position']
        
        # Pattern-based evaluation
        pattern_score = 0
        
        # Check for threatening patterns in all directions
        for r in range(rows):
            for c in range(cols - 3):
                window = [board[r][c+i] for i in range(4)]
                pattern_score += self.evaluate_window(window)
        
        for r in range(rows - 3):
            for c in range(cols):
                window = [board[r+i][c] for i in range(4)]
                pattern_score += self.evaluate_window(window)
        
        for r in range(rows - 3):
            for c in range(cols - 3):
                window = [board[r+i][c+i] for i in range(4)]
                pattern_score += self.evaluate_window(window)
        
        for r in range(3, rows):
            for c in range(cols - 3):
                window = [board[r-i][c+i] for i in range(4)]
                pattern_score += self.evaluate_window(window)
        
        # Apply pattern weight multiplier
        pattern_score *= weights['pattern']
        
        # Look for trap setups (two ways to win)
        trap_score = self.detect_traps(board) * weights['trap']
        
        # Total score combines all components
        total_score = position_score + pattern_score + trap_score
        
        return total_score
    
    def get_phase_weights(self):
        """
        Return the appropriate weight ratios based on the game phase
        """
        # Determine game phase based on move count
        if self.move_count < 10:  # Early game
            return self.weight_ratios['early']
        elif self.move_count < 20:  # Mid game
            return self.weight_ratios['mid']
        else:  # Late game
            return self.weight_ratios['late']
    
    def evaluate_window(self, window):
        """
        Basic window evaluation that considers different patterns
        """
        player = self.position
        opponent = self.opponent.position
        
        # Count pieces in the window
        player_count = window.count(player)
        opponent_count = window.count(opponent)
        empty_count = window.count(0)
        
        # If window has both player pieces and opponent pieces, it's blocked
        if player_count > 0 and opponent_count > 0:
            return 0
        
        # Advanced scoring (applies dynamic weights in evaluate function)
        if player_count == 4:
            return 100  # Immediate win
        elif player_count == 3 and empty_count == 1:
            return 15  # Three in a row with an empty space (potential win)
        elif player_count == 2 and empty_count == 2:
            return 5  # Two in a row with two empty spaces (building)
        elif player_count == 1 and empty_count == 3:
            return 1  # One piece with three empty spaces (future potential)
        
        # Defensive scoring
        if opponent_count == 3 and empty_count == 1:
            return -15  # Block opponent's three in a row (critical)
        elif opponent_count == 2 and empty_count == 2:
            return -3  # Block opponent's two in a row (preventive)
            
        return 0
    
    def detect_traps(self, board):
        """
        Detect situations where a player can create multiple winning threats
        - Using boundary checks to prevent index errors
        """
        rows, cols = board.shape
        trap_count = 0
        
        # Check for multiple winning paths
        for r in range(rows):
            for c in range(cols):
                if board[r][c] == 0:  # Empty cell
                    threat_count = 0
                    
                    # Temporarily place our piece
                    board[r][c] = self.position
                    
                    # Check for multiple winning paths (3-in-a-row with open ends)
                    # Horizontal - with proper boundary checks
                    if c <= cols - 4:
                        window = [board[r][c+i] for i in range(4)]
                        if window.count(self.position) == 3 and window.count(0) == 1:
                            threat_count += 1
                            
                    # Vertical - with proper boundary checks
                    if r <= rows - 4:
                        window = [board[r+i][c] for i in range(4)]
                        if window.count(self.position) == 3 and window.count(0) == 1:
                            threat_count += 1
                            
                    # Diagonal (positive slope) - with proper boundary checks
                    if r <= rows - 4 and c <= cols - 4:
                        window = [board[r+i][c+i] for i in range(4)]
                        if window.count(self.position) == 3 and window.count(0) == 1:
                            threat_count += 1
                            
                    # Diagonal (negative slope) - with proper boundary checks
                    if r >= 3 and c <= cols - 4:
                        window = [board[r-i][c+i] for i in range(4)]
                        if window.count(self.position) == 3 and window.count(0) == 1:
                            threat_count += 1
                    
                    # Undo move
                    board[r][c] = 0
                    
                    # If multiple threats from one move, it's a trap setup
                    if threat_count >= 2:
                        trap_count += 1
        
        return trap_count
    
    def find_immediate_win(self, env, player):
        """
        Check if there's an immediate winning move for the given player
        With proper boundary checks to prevent index errors
        """
        # Get legal moves
        possible = env.topPosition >= 0
        indices = []
        for i, p in enumerate(possible):
            if p: 
                indices.append(i)
                
        # Try center column first as it often leads to more wins
        ordered_indices = sorted(indices, key=lambda x: abs(x - env.shape[1]//2))
                
        # Check each move to see if it results in a win
        for move in ordered_indices:
            row = env.topPosition[move]
            
            # Skip if column is full
            if row < 0:
                continue
                
            # Make the move
            env.board[row][move] = player
            
            # Simple win check (horizontal)
            count = 0
            for c in range(max(0, move-3), min(move+4, env.shape[1])):
                if 0 <= c < env.shape[1]:  # Ensure in bounds
                    if env.board[row][c] == player:
                        count += 1
                    else:
                        count = 0
                    if count >= 4:
                        env.board[row][move] = 0  # Undo move
                        return move
                    
            # Simple win check (vertical)
            count = 0
            for r in range(max(0, row-3), min(row+4, env.shape[0])):
                if 0 <= r < env.shape[0]:  # Ensure in bounds
                    if env.board[r][move] == player:
                        count += 1
                    else:
                        count = 0
                    if count >= 4:
                        env.board[row][move] = 0  # Undo move
                        return move
                    
            # Simple win check (diagonal - positive slope)
            for offset in range(4):
                valid = True
                for i in range(4):
                    if not (0 <= row+offset-i < env.shape[0] and 0 <= move+offset-i < env.shape[1]):
                        valid = False
                        break
                
                if valid and all(env.board[row+offset-i][move+offset-i] == player for i in range(4)):
                    env.board[row][move] = 0  # Undo move
                    return move
            
            # Simple win check (diagonal - negative slope)
            for offset in range(4):
                valid = True
                for i in range(4):
                    if not (0 <= row-offset+i < env.shape[0] and 0 <= move+offset-i < env.shape[1]):
                        valid = False
                        break
                        
                if valid and all(env.board[row-offset+i][move+offset-i] == player for i in range(4)):
                    env.board[row][move] = 0  # Undo move
                    return move
            
            # Undo the move
            env.board[row][move] = 0
                
        return None

# Defining Constants
SQUARESIZE = 100
BLUE = (0,0,255)
BLACK = (0,0,0)
P1COLOR = (255,0,0)
P2COLOR = (255,255,0)

ROW_COUNT = 6
COLUMN_COUNT = 7

pygame.init()

SQUARESIZE = 100

width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT+1) * SQUARESIZE

size = (width, height)

RADIUS = int(SQUARESIZE/2 - 5)

screen = pygame.display.set_mode(size)