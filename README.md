# SmartProcessAnalytics

Smart Process Analytics (SPA) is a Python software for predictive modeling. It is associated with the paper ["Smart Process Analytics" by Weike Sun and Richard D. Braatz](https://doi.org/10.1016/j.compchemeng.2020.107134).

The [Smart\_Process\_Analytics.py](Smart_Process_Analytics.py) file is the main function, which provides the default predictive modeling and the final, evaluated model. All the other files needed to run [Smart\_Process\_Analytics.py](Smart_Process_Analytics.py) are stored in [Code-SPA](Code-SPA). To run SPA on your computer, simply download the [Code-SPA](Code-SPA) folder.

If you want to reset the default hyperparameters in the [Smart\_Process\_Analytics.py](Smart_Process_Analytics.py) file or use other functionalities (e.g., model fitting for multiple times series data set using one model), you can use the data interrogation, model construction and residual analysis by yourself. One example is provided in the file [Example\_1.py](Example/Example_\1.py), which uses different methods directly called from the [cv\_final.py](Code-SPA/cv_final.py) file using the 3D printer example data in the original paper.

The major files in SPA are:
1. [dataset\_property\_new.py](Code-SPA/dataset_property_new.py): functions for data interrogation.
2. [cv\_final.py](Code-SPA/cv_final.py) / [cv\_final\_onestd.py](Code-SPA/cv_final_onestd.py): model construction using different cross-validation strageries (or cross-validation with one standard error rule) for models in SPA.
3. [IC.py](Code-SPA/IC.py): model construction using information criteria for dynamic models.
4. [regression\_models.py](Code-SPA/regression_models.py) / [nonlinear\_regression\_other.py](Code-SPA/nonlinear_regression_other.py): basic linear/nonlinear and DALVEN regression models in SPA.
5. [timeseries\_regression\_RNN.py](Code-SPA/timeseries_regression_RNN.py): RNN model (including training/testing for single/multiple training sets).
6. [timeseries\_regression\_matlab](Code-SPA/timeseries_regression_matlab.py): MATLAB SS model (including training/testing for single/multiple training sets).
7. [timeseries\_regression\_ADAPTx](Code-SPA/timeseries_regression_Adaptx.py): ADAPTx SS-CVA model (including training/testing for single/multiple training sets).

The final result is stored in `selected_model` (for the name that the software picked or the user specified) and `fitting_result`.

Note: MATLAB is required to use the linear dynamic model. It is called through matlab.engine (https://www.mathworks.com/help/matlab/matlab_external/call-matlab-functions-from-python.html). If the user has ADAPTx, it can also be used to create a linear state-space model through SPA. SPA will ask for the url of your ADAPTx package.

Please contact Richard Braatz at braatz@mit.edu for any inquiries.

