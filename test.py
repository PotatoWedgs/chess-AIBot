# Import dependencies
from io import StringIO
import torch 
import pandas as pd
from torch import nn
import torch
import numpy as np
import chess
import chess.pgn

combinations = [(start, end) for start in range(1, 65) for end in range(1, 65) if start != end]     #   Saving the combinations of square to square chess moves

def convert(username, pgn):

    #   Setting up the chess game
    game = chess.pgn.read_game(pgn)
    board = game.board()

    board_dict = {}     #   The dictionary we will use to append to the output of the conversion   
    moves = []      #   The list to save the square moves
    index_moves = []    #   The list to save the index square moves

    user_colour = None     #   The user we are training to be like is by getting their colour

    if game.headers['White'] == username:
        user_colour = True      #   White
    elif game.headers['Black'] == username:
        user_colour = False     #   Black

    for number, move in enumerate(game.mainline_moves()):   #   A for loop over the moves of the game and its move number
        move_dict = {}      #   The dictionary that will store the location of each piece when a move occurs

        if board.turn != user_colour:   #   IF its not the users turn to move, we append the previous move (users move) to the moves list
            moves.append((int(move.from_square)+1, int(move.to_square)+1))

        #   Playing through all the moves of both players for simulation
        board.push(move)

        #   The map filled with the location of each piece on the board
        boardmap = board.piece_map() 

        if board.turn == user_colour:   #   IF its users turn to play a move

            #   The counters for each piece type to keep track of how many of each piece type is on the board
            Wqueen_counter = 1
            Wrook_counter = 1
            Wbishop_counter = 1
            Wknight_counter = 1
            Wpawn_counter = 1

            Bqueen_counter = 1
            Brook_counter = 1
            Bbishop_counter = 1
            Bknight_counter = 1
            Bpawn_counter = 1


            for element in boardmap:      #   A for loop going over the contents in the board map

                #   ALL these if statements are to record where the pieces are located at
                if str(boardmap[element]) == 'K':          
                    move_dict['WKing'] = element+1


                if str(boardmap[element]) == 'Q':            
                    move_dict[f'WQueen{Wqueen_counter}'] = element+1
                    Wqueen_counter += 1


                if str(boardmap[element]) == 'R':               
                    move_dict[f'WRook{Wrook_counter}'] = element+1
                    Wrook_counter += 1


                if str(boardmap[element]) == 'B':                
                    move_dict[f'WBishop{Wbishop_counter}'] = element+1
                    Wbishop_counter += 1


                if str(boardmap[element]) == 'N':
                    move_dict[f'WKnight{Wknight_counter}'] = element+1
                    Wknight_counter += 1


                if str(boardmap[element]) == 'P':
                    move_dict[f'WPawn{Wpawn_counter}'] = element+1
                    Wpawn_counter += 1


                if str(boardmap[element]) == 'k':
                    move_dict['BKing'] = element+1


                if str(boardmap[element]) == 'q':
                    move_dict[f'BQueen{Bqueen_counter}'] = element+1
                    Bqueen_counter += 1


                if str(boardmap[element]) == 'r':
                    move_dict[f'BRook{Brook_counter}'] = element+1
                    Brook_counter += 1


                if str(boardmap[element]) == 'b':
                    move_dict[f'BBishop{Bbishop_counter}'] = element+1
                    Bbishop_counter += 1


                if str(boardmap[element]) == 'n':
                    move_dict[f'BKnight{Bknight_counter}'] = element+1
                    Bknight_counter += 1


                if str(boardmap[element]) == 'p':
                    move_dict[f'BPawn{Bpawn_counter}'] = element+1
                    Bpawn_counter += 1

            #   Saving the whole game of the user's moves in a dictonary
            board_dict[number+3] = move_dict

    #   List to keep note of the columns in the data frame
    columns = []

    df = pd.DataFrame(board_dict).fillna(0)     #   Setting up the dataframe with the numbered moves game

    df = df.transpose()     #   Transposing the data to have our pieces on the columns

    #   Iterating over the columns to append to the columns list
    for col in df.columns:  
        columns.append(col)

    remaining_cols = 96 - len(df.columns)   #   The remaining columns needed to fill to 96

    if remaining_cols > 0:  #   IF there is remaining columns left, we add columns filled with 0s
        for i in range(len(df.columns), len(df.columns) + remaining_cols):
            df['col{}'.format(i+1)] = 0

    #   Iterating over the moves played to create column of indexed moves through possible combinations
    for move in moves:  
        index_moves.append(combinations.index(move))

    return df.to_numpy()

pgn = StringIO(f'[Event "Live Chess"] \n[Site "-"] \n[Date "-"] \n[Round "-"] \n[White "Pak"] \n[Black "-"] \n[Result "0-0"] \n[WhiteElo "0"] \n[BlackElo "0"] \n[TimeControl "0"] \n[EndTime "0"] \n[Termination 0"] \n\n1. d4')

print(convert('Pak', pgn))