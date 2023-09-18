#imports
import logging
import os
import pandas as pd
import numpy as np
import gsflow
#import pyHMT2D

## Framework here:
# - Class: NewCalibration that receives a config file/script
#   - Config file/script specifies

class NewCalibration:

    def __init__(self, model):
        self.algorithm = None
        self.sampling_method = None
        self.obj_funcs = None
        self.modelrun = None
        self.param_adjust = None
        self.model = model()


    #def _create_obj_func_dict(self):

    def MSCUA(self):

    def SUFI2(self):

    def SCEUA(self):

    def nelder_mead(self):

    #def calibrate(self, cal_params, ):

class GSFLOW():
    def __init__(self, control_file_fp, exe_fp):
        self.control_file = control_file_fp
        self.gsflow_exe = exe_fp
        self.gsf_model = gsflow.GsflowModel.load_from_file(self.control_file, self.gsflow_exe)
        self.params = self.gsf_model.prms.parameters
        self.control = self.gsf_model.control
        self.model_mode = self.control.get_record('model_mode')

    def _store_init_params(self, param_names):


    def _change_gsflow_params(self, param_names, new_values):
        if type(param_names) != list:
            param_name = list(param_names)

        for p in param_names:
            if p not in list(self.params):
                raise ValueError('{0} not included in available parameters: {1}'.format(param_name, list(self.params)))
                break

    def _write_params(self):
        self.gsf_model.write_input(write_only=['parameters'])

    def simulations(self):
        self.gsf_model.run_model()
        self.gsf_output = self.gsf_model.get_StatVar()

    def calibrate(self):

#class HECRAS():

#class RiverWare():

#class MODSIM():


def f1(x):
    y = x**2 + 40
    return y

def f2(x):
    y = x**2 + x*4 - 40
    return y

funcs = {
    'f1': f1,
    'f2': f2,
}

class Parent:
    def __init__(self, obj, name, age):
        self.name = name
        self.age = age
        self.output = None
        self.C = obj

    def f1(self, x):
        x2 = f2(x)
        y = x2 ** 2 + 40
        self.output = y

class Child:
    def __init__(self, color, shape):
        self.color = color
        self.shape = shape

    def f2(self, y):
        y = x**2 + x*4 - 40
        return y














if __name__ == '__main__':
    PRMSexe = os.path.join("C:\\Users\\CNB968", "OneDrive - MT", "Modeling", "GSFLOW", "gsflow_2.2.0", "bin", "gsflow.exe")
    Lolocf = os.path.join("C:\\Users\\CNB968", "OneDrive - MT", "Documents", "ArcGIS", "Projects", "Lolo_Cr", "GSFLOW Model", "PRMSonly_test1", "model", "prms", "Lolo_prms.control")
