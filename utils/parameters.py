from keras.optimizers import Adam


# Parameters
class Parameters:
    # Parameters
    batch_size: int = 32
    epochs: int = 5

    hidden_layers: [int] = [300, 225, 225, 225]

    activation_output: str = "softmax"
    activation_hidden: str = "relu"
    loss_function: str = "categorical_crossentropy"

    learning_rate: float = 1e-3
    optimizer = Adam(learning_rate=1e-3)

    train_size: float = 0.8

    board_size: int = 6 * 7
    input_size: int = 6 * 7
    output_size: int = 7  # or 3

    # Validation
    iterations = 100
    monte_carlo_depth = 10
