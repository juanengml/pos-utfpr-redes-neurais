from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


class DataTransformer:
    def __init__(self, df):
        self.df = df

    def drop_irrelevant_columns(self):
        self.df = self.df.drop(
            ['id', 'date', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long'], axis=1
        )
        return self.df

    def plot_correlation_matrix(self):
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            self.df.corr(numeric_only=True),
            annot=False,
            cmap=sns.cubehelix_palette(as_cmap=True)
        )
        plt.title('Mapa de correlação das variáveis selecionadas do dataset')
        plt.show()

    def split_features_and_target(self, target_column='price'):
        x = self.df.drop(target_column, axis=1)
        y = self.df[target_column]
        maxY = y.max()
        y = y/maxY
        print(maxY)  
        scaler = StandardScaler().fit(x)
        x = scaler.transform(x)
        return x, y

    def split_train_test(self, x,y, test_size=0.30):
        trainX, testX, trainY, testY = train_test_split(x, y, test_size=0.30)
        return trainX, testX, trainY, testY