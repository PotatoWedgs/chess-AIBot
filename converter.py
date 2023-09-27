import chess
import chess.pgn
import pandas as pd

def convert(pgn, csv):
    game = chess.pgn.read_game(open(pgn))
    board = game.board()

    board_dict = {}
    moves = []
    index_moves = []

    combinations = [(start, end) for start in range(1, 65) for end in range(1, 65) if start != end]

    for number, move in enumerate(game.mainline_moves()):
        
        move_dict = {}
        moves.append((int(move.from_square)+1, int(move.to_square)+1))

        board.push(move)
        boardmap = board.piece_map() 

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

        for i in boardmap:

            if str(boardmap[i]) == 'K':          
                move_dict['WKing'] = i+1


            if str(boardmap[i]) == 'Q':            
                move_dict[f'WQueen{Wqueen_counter}'] = i+1
                Wqueen_counter += 1


            if str(boardmap[i]) == 'R':               
                move_dict[f'WRook{Wrook_counter}'] = i+1
                Wrook_counter += 1


            if str(boardmap[i]) == 'B':                
                move_dict[f'WBishop{Wbishop_counter}'] = i+1
                Wbishop_counter += 1


            if str(boardmap[i]) == 'N':
                move_dict[f'WKnight{Wknight_counter}'] = i+1
                Wknight_counter += 1


            if str(boardmap[i]) == 'P':
                move_dict[f'WPawn{Wpawn_counter}'] = i+1
                Wpawn_counter += 1


            if str(boardmap[i]) == 'k':
                move_dict['BKing'] = i+1


            if str(boardmap[i]) == 'q':
                move_dict[f'BQueen{Bqueen_counter}'] = i+1
                Bqueen_counter += 1


            if str(boardmap[i]) == 'r':
                move_dict[f'BRook{Brook_counter}'] = i+1
                Brook_counter += 1


            if str(boardmap[i]) == 'b':
                move_dict[f'BBishop{Bbishop_counter}'] = i+1
                Bbishop_counter += 1


            if str(boardmap[i]) == 'n':
                move_dict[f'BKnight{Bknight_counter}'] = i+1
                Bknight_counter += 1


            if str(boardmap[i]) == 'p':
                move_dict[f'BPawn{Bpawn_counter}'] = i+1
                Bpawn_counter += 1

        board_dict[number+1] = move_dict

    columns = []

    df = pd.DataFrame(board_dict).fillna(0)

    df = df.transpose()

    for col in df.columns:
        columns.append(col)

    remaining_cols = 96 - len(df.columns)

    if remaining_cols > 0:
        for i in range(len(df.columns), len(df.columns) + remaining_cols):
            df['col{}'.format(i)] = 0

    for move in moves:
        index_moves.append(combinations.index(move))

    df['move_range_num'] = index_moves

    df.to_csv(csv, index=False)

convert('chess_data/pgn/game1.pgn', 'chess_data/csv/game1.csv')