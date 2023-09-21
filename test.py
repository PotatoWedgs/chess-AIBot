import chess
import chess.pgn
import pandas as pd

names = ["BRook1", "BKnight1", "BBishop1", "BKing", "BQueen1", "BBishop2", "BKnight2", "BRook2",
        "BPawn1", "BPawn2", "BPawn3", "BPawn4", "BPawn5", "BPawn6", "BPawn7", "BPawn8",
        "WPawn1", "WPawn2", "WPawn3", "WPawn4", "WPawn5", "WPawn6", "WPawn7", "WPawn8",
        "WRook1", "WKnight1", "WBishop1", "WKing", "WQueen1", "WBishop2", "WKnight2", "WRook2",
        "WQueen2", "WQueen3", "WQueen4", "WQueen5", "WQueen6", "WQueen7", "WQueen8", "WQueen9",
        "BQueen2", "BQueen3", "BQueen4", "BQueen5", "BQueen6", "BQueen7", "BQueen8", "BQueen9",
        "WRook3", "WRook4", "WRook5", "WRook6", "WRook7", "WRook8", "WRook9", "WRook10",
        "BRook3", "BRook4", "BRook5", "BRook6", "BRook7", "BRook8", "BRook9", "BRook10",
        "WBishop3", "WBishop4", "WBishop5", "WBishop6", "WBishop7", "WBishop8", "WBishop9", "WBishop10",
        "BBishop3", "BBishop4", "BBishop5", "BBishop6", "BBishop7", "BBishop8", "BBishop9", "BBishop10",
        "WKnight3", "WKnight4", "WKnight5", "WKnight6", "WKnight7", "WKnight8", "WKnight9", "WKnight10",
        "BKnight3", "BKnight4", "BKnight5", "BKnight6", "BKnight7", "BKnight8", "BKnight9", "BKnight10"]

def convert(pgn, csv):
    game = chess.pgn.read_game(open(pgn))
    board = game.board()

    board_dict = {}

    for number, move in enumerate(game.mainline_moves()):
        
        move_dict = {}

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

    #df['san_moves'] = raw_list_moves(pgn)

    for name in names:
        if name in columns:
            df[f'{name}diff'] = df[name] - df[name].shift(1)
        


    df.to_csv(csv)


convert('chess_data/pgn/chess.pgn', 'chess_data/csv/chess.csv')