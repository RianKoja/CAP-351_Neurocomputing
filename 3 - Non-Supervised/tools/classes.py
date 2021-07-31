
import minisom

class SomClass:
    def __init__(self, x, y, input_len, sigma, learning_rate, df_name):
        self.x = x
        self.y = y
        self.input_len = input_len
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.df_name = df_name
        self.scaler = None
        self.topographic_error = None
        self.genres_df = None
        self.name = f'{df_name}_som_x{x}_y{y}_sigma{sigma}_lr{learning_rate}'
        self.som = minisom.MiniSom(x=x, y=y, input_len=input_len, sigma=sigma, learning_rate=learning_rate, random_seed=123456)
