# Import dependencies
from io import StringIO
import torch 
import pandas as pd
from torch import nn
import torch
import numpy as np
import train
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


if __name__ == "__main__": 
    #with open('current_game.pgn', 'w') as current_game_pgn:
    with torch.no_grad():   #   Turning off back propogation
        board2 = chess.Board()

        username = 'Potato'
        username_color = 'w'

        moves = []
        move_count = 1
        square_index = ['', 'a1', 'b1', 'c1', 'd1', 'e1', 'f1', 'g1', 'h1', 'a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2', 'h2', 'a3', 'b3', 'c3', 'd3', 'e3', 'f3', 'g3', 'h3', 'a4', 'b4', 'c4', 'd4', 'e4', 'f4', 'g4', 'h4', 'a5', 'b5', 'c5', 'd5', 'e5', 'f5', 'g5', 'h5', 'a6', 'b6', 'c6', 'd6', 'e6', 'f6', 'g6', 'h6', 'a7', 'b7', 'c7', 'd7', 'e7', 'f7', 'g7', 'h7', 'a8', 'b8', 'c8', 'd8', 'e8', 'f8', 'g8', 'h8']
        moves_str = ''

        bool_color = True

        if username_color == "b":
            WHITE = "---"
            BLACK = username
            bool_color = False

        elif username_color == "w":
            WHITE = username
            BLACK = "---"
            bool_color = True

        new_model = train.ChessNeuralNetwork().cuda()
        new_model.load_state_dict(torch.load('model.pt'))

        while board2.is_checkmate() == False:

            if board2.turn == bool_color:
                player_move = input('Play move (SAN): ')

                if board2.turn == True:     #   Black
                    moves.append(f'{move_count}... {player_move}' + ' {[%clk 0:00:00.00]}')

                    move_count += 1

                elif board2.turn == False:
                    moves.append(f'{move_count}. {player_move}' + ' {[%clk 0:00:00.00]}')


                board2.push_san(player_move)
            
                moves_str = str(moves).replace("[", "")
                moves_str = moves_str.replace("]", "")
                moves_str = moves_str.replace("'", "")
                moves_str = moves_str.replace(",", "")

                pgn = StringIO(f'[Event "Live Chess"] \n[Site "-"] \n[Date "-"] \n[Round "-"] \n[White "{WHITE}"] \n[Black "{BLACK}"] \n[Result "0-0"] \n[WhiteElo "0"] \n[BlackElo "0"] \n[TimeControl "0"] \n[EndTime "0"] \n[Termination 0"] \n\n{moves_str}')


            else:
                #try:
                
                moves_str = str(moves).replace("[", "")
                moves_str = moves_str.replace("]", "")
                moves_str = moves_str.replace("'", "")
                moves_str = moves_str.replace(",", "")

                pgn = StringIO(f'[Event "Live Chess"] \n[Site "-"] \n[Date "-"] \n[Round "-"] \n[White "{WHITE}"] \n[Black "{BLACK}"] \n[Result "0-0"] \n[WhiteElo "0"] \n[BlackElo "0"] \n[TimeControl "0"] \n[EndTime "0"] \n[Termination 0"] \n\n{moves_str}')

                data = convert(username, pgn)
                data_tensor = torch.FloatTensor(data)

                print(data)

                """except:
                    print('no')
                    data = np.array([[64.0,63.0,62.0,61.0,60.0,59.0,58.0,57.0,56.0,55.0,54.0,53.0,52.0,50.0,49.0,35.0,16.0,15.0,14.0,13.0,12.0,11.0,10.0,9.0,8.0,7.0,6.0,5.0,4.0,3.0,2.0,1.0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
                    data_tensor = torch.FloatTensor(data)"""
                            
                for iteration, tensor_data in enumerate(data_tensor):    

                    y_val = new_model.forward(tensor_data.cuda())  #   Evaluating the test dataset

                    start_square = combinations[y_val.argmax().item()][0]
                    end_square = combinations[y_val.argmax().item()][1]

                    uci = chess.Move.from_uci(f'{square_index[start_square]}{square_index[end_square]}')
                    bot_move = board2.san(uci)


                    if board2.turn == True:     #   Black
                        moves.append(f'{move_count}... {bot_move}' + ' {[%clk 0:00:00.00]}')

                        move_count += 1

                    elif board2.turn == False:
                        moves.append(f'{move_count}. {bot_move}' + ' {[%clk 0:00:00.00]}')


                    board2.push_san(bot_move)

                    print(bot_move)

                    pgn = StringIO(f'[Event "Live Chess"] \n[Site "-"] \n[Date "-"] \n[Round "-"] \n[White "{WHITE}"] \n[Black "{BLACK}"] \n[Result "0-0"] \n[WhiteElo "0"] \n[BlackElo "0"] \n[TimeControl "0"] \n[EndTime "0"] \n[Termination 0"] \n\n{moves_str}')

                    #moves_str = str(moves).replace("[", "")
                    #moves_str = moves_str.replace("]", "")
                    #moves_str = moves_str.replace("'", "")
                    #moves_str = moves_str.replace(",", "")

                #pgn = f'[Event "Live Chess"] \n[Site "-"] \n[Date "-"] \n[Round "-"] \n[White "{WHITE}"] \n[Black "{BLACK}"] \n[Result "0-0"] \n[WhiteElo "0"] \n[BlackElo "0"] \n[TimeControl "0"] \n[EndTime "0"] \n[Termination 0"] \n\n{moves_str}'
                #pgn = StringIO(pgn)




                """if y_val.argmax().item() == y_test[iteration]:  #   IF models chess move aligns with the test data
            correct += 1
            print(f'{iteration+1}.) {str(y_val.argmax().item())},       {y_test[iteration]},    Correct')
        else:
            print(f'{iteration+1}.) {str(y_val.argmax().item())},       {y_test[iteration]},    Wrong')

    print(f'We got {correct}/{len(X_test)} correct')""" 