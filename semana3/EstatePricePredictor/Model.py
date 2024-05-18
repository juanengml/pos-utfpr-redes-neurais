from sklearn.neural_network import MLPRegressor
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd

class ModelRegressor:
    def __init__(self):
        self.model = MLPRegressor(
            hidden_layer_sizes=(50), activation='logistic', solver='adam',
            max_iter=5000, tol=0.0000001, momentum=0.8, early_stopping=True, epsilon=1e-08,
            n_iter_no_change=10, random_state = 12)

    def train(self, train_x, train_y):
        self.model.fit(train_x, train_y)

    def predict(self, test_x):
        return self.model.predict(test_x)

    def get_n_outputs(self):
        return self.model.n_outputs_

    def plot_loss_curve(self):
        plt.plot(self.model.loss_curve_)
        plt.title("Curva de Perda no Treinamento", fontsize=14)
        plt.xlabel('Épocas')
        plt.ylabel('Custo')
        plt.show()

    def evaluate(self, test_y, y_pred):
        print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(test_y, y_pred))
        print('Mean Squared Error (MSE):', metrics.mean_squared_error(test_y, y_pred))
        print('Root Mean Squared Error (RMSE):', metrics.mean_squared_error(test_y, y_pred, squared=False))
        print('Mean Absolute Percentage Error (MAPE):', metrics.mean_absolute_percentage_error(test_y, y_pred))
        print('R2:', metrics.r2_score(test_y, y_pred))

    def plot_predictions(self, test_y, y_pred, num_samples=40):
        df_temp = pd.DataFrame({'Desejado': test_y, 'Estimado': y_pred}).head(num_samples)
        df_temp.plot(kind='bar', figsize=(10, 6))
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='blue')
        plt.show()

