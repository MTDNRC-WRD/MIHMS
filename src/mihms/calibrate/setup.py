from pathlib import Path
from typing import Union, Optional

import tqdm
import spotpy

from mihms.calibrate.database import MultiDimDb

# The following is a template for an Mscua Setup class, this should be modified to connect your specific model to the
#   calibration scheme.
# The class name is not really important, only that it contains the following methods (which will be unique to your model
#   setup) and attributes.
class MscuaSetup:
    # Define parameters here (before __init__) as variables
    # These are the parameters specific to your model and will need to be defined for each MscuaSetup class
    coef = spotpy.parameter.Uniform(low=0.009, high=0.15)
    exp = spotpy.parameter.Uniform(low=1.0, high=4.0)

    def __init__(self, objective_funcs, dbase: Optional[Union[str, Path, MultiDimDb]] = None):
        # define the multidimensional parameter space here
        # if using spatial data this would be the number of features you are calibrating
        self.parameter_dimension = 0
        # This customizes the dimension name that is carried through the database, here the default is "model" but this
        #   could be feature, model, field, HRU, whatever descriptive name the parameter is attached to.
        self.param_dim_name = 'model'
        # Here you could define any number of attributes that are specific to your model and may be used in the
        #   required methods below, for example observation data.
        self.observation_data = None
        # This can be left alone, this just ensures multiple objective functions can be passed during setup or treated as an iterable
        if isinstance(objective_funcs, list):
            self.objfuncs = objective_funcs
        else:
            self.objfuncs = [objective_funcs]
    # This method must be in the setup class, it is called during calibration to run the model, its only input should be a set
    #   of parameters. This could be an ordered list, tuple, however you want to set it up. It could be a dictionary also
    #   or a parameter file that is edited each simulation. This should return model output...for more complex model
    #   setups you might have a lot of code here to ingest parameters, change the parameters in the model files, and process the model outputs
    #   so that they can be returned as a Python object.
    def simulation(self, params):
        model_out = None
        return model_out

    # This is a required method. It can be more complex if needed but it doesn't really need any arguments, it just needs to
    #   return the observation data to compare with the modeled data. You could define your observations within this
    #   method or load/retrieve them from their source as well.
    def evaluation(self):
        return self.observation_data

    # This must return a dictionary where the keys are the objective function names
    # This can be customized to whatever you need for your model, the multidimensional objective functions in the
    #   obj_funcs module can be manipulated by passing the axis argument (calculated objective function on a particular
    #   axis or for the entire array across all axes). They also include an option to return a dictionary with the
    #   objective function name (default is to just return numpy array), this dictionary is necessary for tracking results
    #   through the algorithm for each objective function.
    def objectivefunction(self, observation, simulation):
        results = {}
        for o in tqdm.tqdm(self.objfuncs, desc="Calculating Objective Functions", leave=False):
            r = o(observation=observation, simulation=simulation, return_dict=True, axis=1)
            results.update(r)

        return results