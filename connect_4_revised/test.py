from connect4 import connect4
from players import randomAI, alphaBetaAI
from montecarlo import monteCarloAI


board_shape = (6,7)
visualize = False 
verbose = False
limit_players = (1,2)
time_limit = (3.0, 3.0)
cvd_mode = False

def play_game(p1, p2):
    '''
    p1 - connect4Player - player who will play first 
    p2 - connect4Player - player who will play second 

    returns winner
    '''

    c4 = connect4(
        player1 = p1, 
        player2 = p2, 
        board_shape = board_shape,
        visualize = visualize,
        limit_players = limit_players,
        time_limit = time_limit,
        verbose = verbose,
        save = False,
        CVDMode = False,
        print_time_logs= True
        ) 

    winner = c4.play()

    return winner == 0, winner == 1, winner == 2

w = 0 
t = 0 
l = 0

for competitor, n_trials in zip([randomAI, monteCarloAI], [5, 5]): 
    for i in range(n_trials):
        
        # You play first
        t1, w1, l1 = play_game(alphaBetaAI(1, i), competitor(2, i))
        # You play second
        t2, l2, w2 = play_game(competitor(1, i), alphaBetaAI(2, i))

        # Track metrics
        w += w1 + w2
        t += t1 + t2
        l += l1 + l2

        print(f"Competitor: {competitor} Test: {i} W: {w1 + w2} T: {t1 + t2} L: {l1 + l2}" )

# Print Metrics
# wins are worth 1pt and ties are worth 0.5pts
points = w + t * 0.5

print(f"Wins: {w} | Ties: {t} | Losses: {l} | Points: {points}/20")



