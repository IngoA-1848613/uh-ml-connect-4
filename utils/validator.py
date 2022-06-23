import numpy as np
from sklearn.model_selection import KFold
from lib.connectfour import Game

PLAYER_RANDOM = -1
PLAYER_AI = 1
DRAW = 0


class Validator:

    @staticmethod
    def validate_against_game(predict, n=0, iterations=100, disable_print=False):
        result_values = {PLAYER_AI: "ai", PLAYER_RANDOM: "random", DRAW: "draw"}
        starts_values = {PLAYER_AI: "ai", PLAYER_RANDOM: "random"}

        results = {PLAYER_AI: 0, PLAYER_RANDOM: 0, DRAW: 0}
        starts = {PLAYER_AI: 0, PLAYER_RANDOM: 0}

        for i in range(iterations):

            game = Game()
            prev_move = -1

            active_player = PLAYER_AI if i < (iterations / 2) else PLAYER_RANDOM
            start_player = active_player

            while game.check_status() is None:
                if active_player == PLAYER_AI:
                    move = predict(game, start_player, active_player, prev_move)
                    prev_move = move
                else:
                    if n == 0:
                        move = game.random_action(legal_only=True)
                    else:
                        move, _ = game.smart_action(player=active_player, n=n, legal_only=True)
                    prev_move = move

                game.play_move(player=active_player, column=move)
                active_player *= -1

            starts[start_player] += 1
            results[game.status] += 1

            results_ai = results[PLAYER_AI]
            results_random = results[PLAYER_RANDOM]
            wr = results_ai / (results_ai + results_random)
            sn = "first" if start_player == PLAYER_AI else "second"
            wn = "wins" if game.status == PLAYER_AI else "loses"
            
            if not disable_print:
                print(f"iteration ({i}): ai goes {sn} and {wn} - win-rate ({wr:.2f})")

        results_ai = results[PLAYER_AI]
        results_random = results[PLAYER_RANDOM]
        results_draw = results[DRAW]
        win_rate = results_ai / (results_ai + results_random)
        if not disable_print:
            print(f"win-rate: {win_rate * 100}% and {results_draw} draws")

        return win_rate

    @staticmethod
    def accuracy(y_truth, y_prediction):
        score = 0

        for i in range(len(y_truth)):
            argmax_truth, argmax_prediction = np.argmax(y_truth[i]), np.argmax(y_prediction[i])

            if argmax_truth == argmax_prediction:
                score += 1

        return (score / len(y_truth)) * 100

    @staticmethod
    def cross_validation(model, x, y):
        splits = 5
        kf = KFold(n_splits=splits, shuffle=True)

        count = 0
        for train_range, test_range in kf.split(x):
            x_train, x_test = x[train_range], x[test_range]
            y_train, y_test = y[train_range], y[test_range]

            model.create_model()
            model.train_model_xy(x_train, y_train)

            y_prediction = model.predict(x_test)
            accuracy = Validator.accuracy(y_test, y_prediction)
            print(f"Accuracy for fold no. {count} is: {accuracy}")

            count += 1

    @staticmethod
    def cross_validation_against_game(model, n=5):
        x, y = model.get_input_output()

        splits = 5
        kf = KFold(n_splits=splits, shuffle=True)

        count = 0
        for train_range, test_range in kf.split(x):
            x_train, x_test = x[train_range], x[test_range]
            y_train, y_test = y[train_range], y[test_range]

            model.create_model()
            model.train_model_xy(x_train, y_train)
            accuracy = Validator.validate_against_game(model.predict_move, n=n, disable_print=True)
            print(f"Accuracy for fold no. {count} is: {accuracy}")

            count += 1
