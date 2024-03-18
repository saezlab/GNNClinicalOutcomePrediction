import numpy as np

#Linear Regression Parameters
rgr_linear_regression_params = dict(
            fit_intercept = [True, False],
            positive = [True, False]
            )

#Support Vector Machine Parameters
rgr_svm_params = dict(
            kernel = ["poly","rbf","sigmoid",'sigmoid'],
            C = np.linspace(0.1,50,100),
            gamma = [10**x for x in np.arange(-5,5,dtype = float)],
            max_iter = np.arange(50,200)
            )

#Random Forest Parameters
rgr_random_forest_params = dict(
            n_estimators = [int(i) for i in np.linspace(10,50,num=40)],
            max_features = ["auto","sqrt","log2"],
            bootstrap = [True, False],
            max_depth = np.arange(1,32),
            min_samples_split = np.linspace(0.1, 1.0, 9, endpoint=True)
            )

#Multilayer Perceptron Parameters
rgr_mlp_params = dict(
            hidden_layer_sizes = ([int(x) for x in np.arange(100,300)],[int(x) for x in np.arange(1,5)]),
            activation = ['logistic', 'tanh', 'relu'],
            solver = ['sgd','adam'],
            alpha = np.linspace(0.0001,0.001,num = 10),
            batch_size = [100,150],
            learning_rate = ['constant', 'invscaling', 'adaptive'],
            max_iter = np.arange(20,50)
            )

#Decision Tree Parameters
rgr_decision_tree_params = dict(
            criterion = ['mse', 'friedman_mse', 'mae'],
            max_features = ["auto", "sqrt", "log2"],
            max_depth = np.arange(1,32),
            min_samples_split = np.linspace(0.1, 1.0, 9, endpoint=True),
            max_leaf_nodes = np.arange(2,20)
            )

#Gradient Boosting Paramters
rgr_gradient_boosting_params = dict(
            loss = ['ls','lad','huber','quantile'],
            learning_rate = np.linspace(0.01,0.15,num = 10),
            n_estimators = np.arange(75,150),
            criterion = ['mse','mae','friedman_mse'],
            max_depth = np.arange(3,10),
            subsample = np.linspace(0.5,1.5,10,endpoint=True)
            )