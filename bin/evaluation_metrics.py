from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import math
print(mean_squared_error(y_true, y_pred))
print(math.sqrt(mean_squared_error(Y_test, Y_predicted)))

print(mean_absolute_error(y_true, y_pred))
r2_score(y_true, y_pred) 