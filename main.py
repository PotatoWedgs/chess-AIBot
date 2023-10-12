# Import dependencies
import torch 
import pandas as pd
from torch import nn
import torch
import numpy as np
from train import *
import chess
import chess.pgn

board = chess.Board()
while board.is_checkmate() == False:
    print(board.legal_moves)

    username = input('Username of the player: ')
    username_color = input('Color of the player: ')
    player_move = input('Play move (SAN): ')

    #chess.Move.from_uci(player_move) in board.legal_moves:
    uci_move = board.push_san(player_move).uci()

    game = chess.pgn.Game()

    if username_color == "black":
        game.headers["White"] = "-"
        game.headers["Black"] = username_color
    elif username_color == "white":
        game.headers["White"] = username_color
        game.headers["Black"] = "-"

    node = game.add_variation(chess.Move.from_uci(uci_move))
    node.comment = "Comment"

print(pgn_string)    

"""

    #   Testing
    with torch.no_grad():   #   Turning off back propogation
        correct = 0
        
        for iteration, tensor_data in enumerate(X_test):    
            y_val = model.forward(tensor_data)  #   Evaluating the test dataset

            if y_val.argmax().item() == y_test[iteration]:  #   IF models chess move aligns with the test data
                correct += 1
                print(f'{iteration+1}.) {str(y_val.argmax().item())},       {y_test[iteration]},    Correct')
            else:
                print(f'{iteration+1}.) {str(y_val.argmax().item())},       {y_test[iteration]},    Wrong')

        print(f'We got {correct}/{len(X_test)} correct')"""