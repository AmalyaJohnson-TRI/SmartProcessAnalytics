# Smart Process Analytics

Smart Process Analytics (SPA) is a Python software for predictive modeling. The original version is associated with the paper ["Smart process analytics for predictive modeling" by Weike Sun and Richard D. Braatz](https://doi.org/10.1016/j.compchemeng.2020.107134). Since 2022, it has been updated by Pedro Seber.

To run SPA on your computer, simply download the [most recent release](https://github.com/PedroSeber/SmartProcessAnalytics/releases). [SPA.py](Code-SPA/SPA.py) comes with default hyperparameters for its models, but all hyperparameters are customizable by the user. To learn how to do so, please read the documentation. You may also check the [Examples](Examples) folder and the [README](Examples/README.md) within.

The major files in SPA are:
1. [SPA.py](Code-SPA/SPA.py): the main file and what should be called by the user. Calls the files below depending on what inputs have been passed by the user or the properties of the data.
2. [cv\_final.py](Code-SPA/cv_final.py): performs cross-validation (or IC calculations) to automatically determine the best hyperparameters. Also trains the final model after validation.
3. [regression\_models.py](Code-SPA/regression_models.py) / [nonlinear\_regression\_other.py](Code-SPA/nonlinear_regression_other.py): called multiple times by cv\_final; runs a model once based on one combination of hyperparameters.
4. [dataset\_property\_new.py](Code-SPA/dataset_property_new.py): functions for data interrogation (whether the data exhibit nonlinearity, multicollinearity, and/or dynamics).
5. [timeseries\_regression\_matlab](Code-SPA/timeseries_regression_matlab.py): MATLAB SS model (including training/testing for single/multiple training sets).
6. [timeseries\_regression\_ADAPTx](Code-SPA/timeseries_regression_Adaptx.py): ADAPTx SS-CVA model (including training/testing for single/multiple training sets).

A typical run of [SPA.py](Code-SPA/SPA.py) automatically calls [cv\_final.py](Code-SPA/cv_final.py) once to determine the best hyperparameters and return the best model. For each hyperparameter, [cv\_final.py](Code-SPA/cv_final.py) automatically calls [regression\_models.py](Code-SPA/regression_models.py) once per hyperparameter combination for validation. If the user has not supplied a model type (or types), [SPA.py](Code-SPA/SPA.py) also calls [dataset\_property\_new.py](Code-SPA/dataset_property_new.py) to determine the most adequate model(s) for the data.

The final result is stored in the `selected_model` and `fitting_result` variables. It is also saved in the form of .json and .p files.

Note: MATLAB is required to use the linear dynamic model. It is called through matlab.engine (https://www.mathworks.com/help/matlab/matlab_external/call-matlab-functions-from-python.html). If the user has ADAPTx, it can also be used to create a linear state-space model through SPA.

Please contact Richard Braatz at braatz@mit.edu for any inquiries.

