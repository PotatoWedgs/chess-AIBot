import chess
import chess.pgn
import pandas as pd
import re

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



chess_squares = {
 'a1': 1, 'a2': 2, 'a3': 3, 'a4': 4, 'a5': 5, 'a6': 6, 'a7': 7, 'a8': 8,
 'b1': 9, 'b2': 10, 'b3': 11, 'b4': 12, 'b5': 13, 'b6': 14, 'b7': 15, 'b8': 16,
 'c1': 17, 'c2': 18, 'c3': 19, 'c4': 20, 'c5': 21, 'c6': 22, 'c7': 23, 'c8': 24,
 'd1': 25, 'd2': 26, 'd3': 27, 'd4': 28, 'd5': 29, 'd6': 30, 'd7': 31, 'd8': 32,
 'e1': 33, 'e2': 34, 'e3': 35, 'e4': 36, 'e5': 37, 'e6': 38, 'e7': 39, 'e8': 40,
 'f1': 41, 'f2': 42, 'f3': 43, 'f4': 44, 'f5': 45, 'f6': 46, 'f7': 47, 'f8': 48,
 'g1': 49, 'g2': 50, 'g3': 51, 'g4': 52, 'g5': 53, 'g6': 54, 'g7': 55, 'g8': 56,
 'h1': 57, 'h2': 58, 'h3': 59, 'h4': 60, 'h5': 61, 'h6': 62, 'h7': 63, 'h8': 64
}



def raw_list_moves(pgn):
    raw_pgn = " ".join([line.strip() for line in open(pgn)])

    comments_marked = raw_pgn.replace("{","<").replace("}",">")
    STRC = re.compile("<[^>]*>")
    comments_removed = STRC.sub(" ", comments_marked)

    STR_marked = comments_removed.replace("[","<").replace("]",">")
    str_removed = STRC.sub(" ", STR_marked)

    MOVE_NUM = re.compile("[1-9][0-9]* *\.")
    just_moves = [_.strip() for _ in MOVE_NUM.split(str_removed)]

    last_move = just_moves[-1]
    RESULT = re.compile("( *1 *- *0 *| *0 *- *1 *| *1/2 *- *1/2 *)")
    last_move = RESULT.sub("", last_move)
    moves = just_moves[:-1] + [last_move]

    x = 0

    listed_moves = []

    for m in moves:

        m = m.removesuffix('#')
        m = m.removesuffix('+') 
        m = m.removesuffix('=Q')
        m = m.removesuffix('=R')
        m = m.removesuffix('=B')
        m = m.removesuffix('N')
        

        if m[:3] == '.. ':
            m = m.removeprefix('.. ')

            if len(m) > 2:
                if m[:1] == 'K':
                    x = -1 + chess_squares[str(m[-2:])]

                elif m[:1] == 'Q':
                    x = 64 + chess_squares[str(m[-2:])]

                elif m[:1] == 'R':
                    x = 129 + chess_squares[str(m[-2:])]

                elif m[:1] == 'B':
                    x = 194 + chess_squares[str(m[-2:])]

                elif m[:1] == 'N':
                    x = 259 + chess_squares[str(m[-2:])]

                elif m == 'O-O':
                    x = -1 + chess_squares['g8']

                elif m == 'O-O-O':
                    x = -1 + chess_squares['c8'] 

                else:
                    x = 324 + chess_squares[str(m[-2:])]

            elif len(m) == 2:
                x = 324 + chess_squares[str(m[-2:])]
        
        else:
            if len(m) > 2:
                if m[:1] == 'K':
                    x = -1 + chess_squares[str(m[-2:])]

                elif m[:1] == 'Q':
                    x = 64 + chess_squares[str(m[-2:])]

                elif m[:1] == 'R':
                    x = 129 + chess_squares[str(m[-2:])]

                elif m[:1] == 'B':
                    x = 194 + chess_squares[str(m[-2:])]

                elif m[:1] == 'N':
                    x = 259 + chess_squares[str(m[-2:])]

                elif m == 'O-O':
                    x = -1 + chess_squares['g1']

                elif m == 'O-O-O':
                    x = -1 + chess_squares['c1'] 

                else:
                    x = 324 + chess_squares[str(m[-2:])]
            
            elif len(m) == 2:
                x = 324 + chess_squares[str(m[-2:])]

        listed_moves.append(x)

    listed_moves.remove(0)

    return listed_moves


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

    df['move_range_num'] = raw_list_moves(pgn)

    df.to_csv(csv, index=False)

convert('chess_data/pgn/chess.pgn', 'chess_data/csv/chess.csv')