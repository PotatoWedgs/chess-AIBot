import chess
import chess.pgn
import pandas as pd
import re


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

    listed_moves = []

    for m in moves:
        if '..' in m:
            m = m.removeprefix('.. ')

        listed_moves.append(m)

    listed_moves.remove('')

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

    df = pd.DataFrame(board_dict).fillna(0)
    
    df = df.transpose()

    remaining_cols = 96 - len(df.columns)

    if remaining_cols > 0:
        for i in range(len(df.columns), len(df.columns) + remaining_cols):
            df['col{}'.format(i)] = 0

    df['san_moves'] = raw_list_moves(pgn)

    df.to_csv(csv)


convert('chess_data/pgn/chess.pgn', 'chess_data/csv/chess.csv')