import chess
import chess.pgn
import pandas as pd

def split_pgn(input_pgn, directory_output, num):
    with open(input_pgn, "r") as file:
        pgn_text = file.read()

    games = pgn_text.split("\n\n[Event ")

    for i, game in enumerate(games):  # Skip the first element (empty string)
        game = "[Event " + game  # Restore the lost tag
        with open(f"{directory_output}game{i + (num*50) + 1}.pgn", "w") as game_file:
            game_file.write(game)


def convert(username, pgn, csv):
    game = chess.pgn.read_game(open(pgn))
    board = game.board()

    board_dict = {}
    moves = []
    index_moves = []

    combinations = [(start, end) for start in range(1, 65) for end in range(1, 65) if start != end]

    user = None

    if game.headers['White'] == username:
        user = True
    elif game.headers['Black'] == username:
        user = False

    for number, move in enumerate(game.mainline_moves()):
        move_dict = {}

        if board.turn != user:
            moves.append((int(move.from_square)+1, int(move.to_square)+1))

        board.push(move)
        boardmap = board.piece_map() 

        if board.turn == user:
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


#for num in range(10):
    #split_pgn(f"chess_data/pgn_packed/chess_com_games_2023-09-28 ({num}).pgn", "chess_data/pgn/", num)

i = 0

while i < 500:
    convert('PotatoWedgesBinAmmo', f'chess_data/pgn/game{i+1}.pgn', f'chess_data/traincsv/game{i+1}.csv')
    i += 1