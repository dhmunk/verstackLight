
# verstackLight.LGBMTuner
## A light version of verstack 3.8.9
#### Light-version creator: Daniel Hans Munk, 2023
#### Original verstack creator: Danil Zherebtsov
---
### Links
* [Original verstack Github repository](https://github.com/DanilZherebtsov/verstack)
* [Original verstack documentation](https://verstack.readthedocs.io/en/latest/)





#### **Importing verstackLight**

```
import sys
ABSOLUTE_verstackLight_FOLDER_PATH = r''
sys.path.append(ABSOLUTE_verstackLight_FOLDER_PATH[:-1].rsplit('/', 1)[0])

from verstackLight import LGBMTuner
```
#### **verstackLight dependencies**
* [optuna](https://optuna.readthedocs.io/en/stable/index.html)
* [lightgbm](https://lightgbm.readthedocs.io/en/latest/index.html)
* [matplotlib](https://matplotlib.org/)
* [plotly](https://plotly.com/python/)
* [scikit-learn](https://scikit-learn.org/stable/)
* [pandas](https://pandas.pydata.org/)
* [numpy](https://numpy.org/)
* [scipy](https://www.scipy.org/)



---
# **LGBMTuner**  - automated lightgbm models tuner with optuna

Fully automated lightgbm model hyperparameter tuning class with optuna under the hood. 
LGBMTuner selects optimal hyperparameters based on executed trials (configurable), optimizes n_estimators and fits the final model to the whole train set.
Feature importances are available in numeric format, as a static plot, and as an interactive plot (html).
Optimization history and parameters importance in static and interactive formats are alse accesable by built in methods.

### Logic
The only required user inputs are the X (features), y (labels) and evaluation metric name, LGBMTuner will handle the rest.

By default LGBMTuner will automatically:
1. Configure various LGBM model hyperparameters for regression or classification based on input data
    - lgbm model type (regression/classification) is inferred from the labels and evaluation metric (passed by user)

    - optimization metric may be different from the evaluation metric (passed by user). LGBMTuner at hyperparameters search stage imploys the error reduction strategy, thus:
    
    - most regression task type metrics are supported for optimization, if not, MSE is selected for optimization
    
    - for classification task types hyperparameters are tuned by optimizing log_loss, n_estimators are tuned with evaluation_metric
    
    - early stopping is engaged at each stage of LGBMTuner optimizations
    
    - for every trial (iteration) a random train_test_split is performed (stratified for classification) eliminating the need for cross-validation
    
    - lgbm model initial parameters!=defaults and are inferred from the data stats and built in logic
    
    - optimization parameters and their search space are inferred from the data stats and built in logic
    
    - LGBMTuner class instance (after optimization) can be used for making predictions with conventional syntaxis (predict/predict_proba)
    
    - verbosity is controlled and by default outputs only the necessary optimization process/results information

2. Optimize the follwoing parameters within the defined ranges:
    - 'feature_fraction' : {'low': 0.5, 'high': 1}
  
    - 'num_leaves' : {'low' : 16, 'high': 255}
  
    - 'bagging_fraction' : {'low' : 0.5, 'high' : 1.0}
    
    - 'min_sum_hessian_in_leaf' : {'low' : 1e-3, 'high' " 10.0}
    
    - 'lambda_l1' : {'low' " 1e-8, 'high' : 10.0}
    
    - 'lambda_l2' : {'low' " 1e-8, 'high' : 10.0}

### *note* 
* User may define other lightgbm parameters and their respective grids for optimization by changing the LGBM.grid dictionary after the class is initialized, please refer to the examples below.

* LGBM categorical_feature is supported. According to [LGBM docs](https://lightgbm.readthedocs.io/en/latest/Parameters.html#categorical_featur). Unique values within each categoric feature must be encoded by consecutive integers and casted to 'categoric' dtype: df['categoric_column'] = df['categoric_column'].astype('categoric') before sending the data to LGBMTuner.fit() method.

* All other LGBM configurations are supported from version 1.1.0. Pass the desired parameters to a `custom_lgbm_params` argument at LGBMTuner init.

### **Initialize LGBMTuner**

```
 from verstackLight import LGBMTuner
  
  # initialize with default parameters
  tuner = LGBMTuner(metric = 'rmse')
  
  # initialize with selected parameters
  tuner = LGBMTuner(metric = 'rmse', 
                    trials = 100, 
                    refit = True, 
                    verbosity = 1, 
                    visualization = True, 
                    seed = 42,
                    device_type = 'cpu',
                    n_jobs = 2)

```


 

Parameters (keyword arguments only)
===========================
* ``metric`` [default=None]

  Evaluation metric for hyperparameters optimization. LGBMTuner supports the following metrics (note the syntax) \
['mae', 'mse', 'rmse', 'rmsle', 'mape', 'smape', 'rmspe', 'r2', 'auc', 'gini', 'log_loss', 'accuracy', 'balanced_accuracy', 'precision', 'precision_weighted', 'precision_macro', 'recall', 'recall_weighted', 'recall_macro', 'f1', 'f1_weighted', 'f1_macro', 'lift']

* ``trials`` [default=100]

  Number of trials to run

* ``refit`` [default=True]

  Fit the model with optimized hyperparameters on the whole train set (required for feature_importances, plot_importances() and prediction methods)

* ``verbosity`` [default=1]

  Console verbosity level: 0 - no output except for optuna CRITICAL errors and builtin exceptions; 
  (1-5) based on optuna.logging options. The default is 1

* ``visualization`` [default=True]

  Automatically output feature_importance & optimization plots into the console after tuning. Plots are also available on demand by corresponding methods

* ``seed`` [default=42]

  Random state parameter

* ``device_type`` [default="cpu"]

  Device for the tree learning, you can use GPU to achieve the faster learning. Acceptable parameters are "cpu", "gpu", "cuda", "cuda_exp"

* ``custom_lgbm_params`` [default={}]

  Any supported LGBM parameters to be set for the model. Please refer to the [LGBM docs](https://lightgbm.readthedocs.io/en/latest/Parameters.html#categorical_featur) for the full list of parameters and their descriptions

* ``eval_results_callback`` [default=None]

  Callback function to be applied on the eval_results dictionary that is being populated with evaluation metric score upon completion of each training trial

* ``n_jobs`` [default=1]

  Number of parallel threads to be used for LGBM model training. If n_jobs=None all CPU cores are used (minus 2) for safeguarding.


Methods
===========================
* ``fit(X, y)``

  Execute LGBM model hyperparameters tuning

    Parameters

    - ``X`` [pd.DataFrame]

      Train features
    
    - ``y`` [pd.Series]
      
      Train labels

    - ``optuna_study_params`` [dict, default=None]

      Optuna study parameters. Please refer to the [Optuna docs](https://optuna.readthedocs.io/en/stable/reference/study.html#optuna.study.Study.optimize) for the full list of parameters and their descriptions

* ``fit_optimized(X, y)``

  Train model with tuned params on whole train data

    - ``X`` [np.array]

      Train features
    
    - ``y`` [np.array]

* ``predict(test, threshold = 0.5)``

  Predict by optimized model on new data

    - ``test`` [pd.DataFrame]

      Test features
    
    - ``threshold`` [default=0.5]

      Classification threshold (applicable for binary classification)

      returns: 
      array of int

* ``predict_proba(test)``

  Predict probabilities by optimized model on new data

    - ``test`` [pd.DataFrame]

      Test features

  returns: array of float

* ``plot_importances(n_features = 15, 
                     figsize = (10,6), 
                     interactive = False, 
                     display = True, 
                     dark = True,
                     save = False,
                     plotly_fig_update_layout_kwargs = {})``

  Plot feature importance
    
    - ``n_features`` [default=15]

      Number of important features to plot

    - ``figsize`` [default=(10,6)]

      plot size

    - ``interactive`` [default=False]

      Create & display with the default browser the interactive html plot or (if browser disply is unavailable) save to current wd.

    - ``display`` [default=True]

      Display plot in browser. If False, plot will be saved in cwd.

    - ``dark`` [default=True]

      Enable dark or light mode for plot.

    - ``save`` [default=False]

      Save plot to current working directory.

    - ``plotly_fig_update_layout_kwargs`` [default={}]

      kwargs for plotly.fig.update_layout() function. The default is empty dict and default_plotly_fig_update_layout_kwargs configured inside the plot_importances() will be used.

* ``plot_optimization_history(interactive = False)``

  Plot optimization function improvement history

    - ``interactive`` [default=False]

      Create & display with the default browser the interactive html plot or (if browser disply is unavailable) save to current wd.

    - ``display`` [default=True]

      Display plot in browser. If False, plot will be saved in cwd.

* ``plot_param_importances(interactive = False)``

  Plot params importance plot
  
    - ``interactive`` [default=False]

      Create & display with the default browser the interactive html plot or (if browser disply is unavailable) save to current wd.

    - ``display`` [default=True]

      Display plot in browser. If False, plot will be saved in cwd.

* ``plot_intermediate_values(interactive = False, legend = False)``

  Plot optimization trials history. Shows successful and terminated trials. If trials > 50 it is better to study the interactive version

    - ``interactive`` [default=False]

      Create & display with the default browser the interactive html plot or (if browser disply is unavailable) save to current wd.

    - ``legend`` [default=False]

      Plot legen on a static plot

    - ``display`` [default=True]

      Display plot in browser. If False, plot will be saved in cwd.

**Attributes**

* ``metric``

  Evaluation metric defined by user at LGBMTuner init

* ``refit``

  Setting for refitting the optimized model on whole train dataset

* ``verbosity``

  Verbosity level settings

* ``visualization``

  Automatic plots output after optimization setting
  
* ``seed``

  Random state value

* ``fitted_model``

  Trained LGBM booster model with optimized parameters

* ``feature_importances``

  Feature importance values

* ``study``

  optuna.study.study.Study object after hyperparameters tuning

* ``init_params``

  initial LGBM model parameters

* ``best_params``

  learned optimized parameters

* ``eval_results``

  dictionary with evaluation results per each of non-pruned trials measured by a function derived from the ``metric`` argument

* ``grid``

  dictionary with all the supported and currently selected optimization parameters

