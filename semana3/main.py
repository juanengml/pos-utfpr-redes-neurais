from EstatePricePredictor.Database import DatasetLoader
from EstatePricePredictor.Transformers import DataTransformer
from EstatePricePredictor.Model import ModelRegressor

def main():
    df = DatasetLoader.load_dataset("kc_house_data.csv")
    transformer = DataTransformer(df)
    transformer.plot_correlation_matrix()
    df = transformer.drop_irrelevant_columns()
    x, y = transformer.split_features_and_target()
    trainX, testX, trainY, testY = transformer.split_train_test(x, y)

    # Train and evaluate model
    model = ModelRegressor()
    model.train(trainX, trainY)
    y_pred = model.predict(testX)
    model.plot_predictions(testY, y_pred)
    model.plot_loss_curve()
    model.evaluate(testY, y_pred)

if __name__ == "__main__":
    main()