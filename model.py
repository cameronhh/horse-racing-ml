from loading import DataManager
from betting import BetManager
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
import pandas as pd
data_manager = DataManager('data/class1_master.csv', 0)

X_train, X_test, y_train, y_test = data_manager.load(train_test_split=0.3)
original_test_data = data_manager.get_original_test_data()

### MODEL - replace with any sklearn model ###
model = MLPRegressor(
        hidden_layer_sizes=(10,5,3), 
        activation="relu",
        solver="lbfgs", 
        max_iter=400, 
        learning_rate="adaptive", 
        verbose=True,
        learning_rate_init=0.001, 
        alpha=0.001, 
        batch_size=200, 
        tol=1e-4, 
        random_state=None,
        early_stopping=False, 
)

model.fit(X_train, y_train)
train_true, train_pred = y_train, model.predict(X_train)
y_true, y_pred = y_test, model.predict(X_test)

### PRINTING METRICS ###
train_mae = metrics.mean_absolute_error(train_true, train_pred)
test_mae = metrics.mean_absolute_error(y_true, y_pred)
print('Training Set MAE:    %f' % train_mae)
print('Test Set MAE:        %f' % test_mae)

### BETMANAGER (requires pandas data types) ###
bet_manager = BetManager()
bet_manager.calculate_betting_outcomes(original_test_data, pd.Series(y_pred, name='predicted_margin'))
