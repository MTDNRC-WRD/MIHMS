from pathlib import Path
from typing import Union, Optional
from datetime import datetime
import json

import tomli
from gsflow import ControlFile, ControlRecord
from gsflow import PrmsParameters

from mihms.config import prep


class ControlBase:

    def __init__(self,
                 start_time: str,
                 end_time: str,
                 data_files: Union[str, Path, list],
                 param_files: Union[str, Path, list, PrmsParameters],
                 output_path: Union[str, Path],
                 file_name: str = 'prms_model',
                 model_mode: str = 'PRMS5',
                 defaults: bool = True
                 ):

        self.start_time = datetime.strptime(start_time, "%Y-%m-%d")
        self.end_time = datetime.strptime(end_time, "%Y-%m-%d")
        self.control_filename = file_name

        if isinstance(output_path, str):
            self.model_wd = Path(output_path) / 'prms_model_output'
        elif isinstance(output_path, Path):
            self.model_wd = output_path / 'prms_model_output'
        else:
            raise ValueError("The model output file path is not recognized as pathlike or does not exist.")

        self.model_wd.mkdir(parents=True, exist_ok=True)

        self.control_obj = ControlFile(records_list=[], name=file_name)

        self.control_obj._records_list.append(ControlRecord("model_mode", [model_mode], datatype=4))
        self.control_obj._records_list.append(ControlRecord(
            "start_time",
            [
                self.start_time.year,
                self.start_time.month,
                self.start_time.day,
                self.start_time.hour,
                self.start_time.minute,
                self.start_time.second
            ],
                  datatype=1
        ))

        self.control_obj._records_list.append(ControlRecord(
            "end_time",
            [
                self.end_time.year,
                self.end_time.month,
                self.end_time.day,
                self.end_time.hour,
                self.end_time.minute,
                self.end_time.second
            ],
                  datatype=1
        ))

        if isinstance(param_files, (str, Path)):
            self.control_obj._records_list.append(ControlRecord("param_file", [param_files], datatype=4))
        elif isinstance(param_files, list):
            self.control_obj._records_list.append(ControlRecord("param_file", param_files, datatype=4))
        elif isinstance(param_files, PrmsParameters):
            parameter_files = param_files.parameter_files
            recl = []
            for f in parameter_files:
                if f is None:
                    recl.append(f"{self.control_filename}.param")
                else:
                    recl.append(f)
            self.control_obj._records_list.append(ControlRecord("param_file", recl, datatype=4))
        else:
            raise ValueError("The supplied parameter files argument is invalid. Must be pathlike, list of paths, or "
                             "pygsflow PrmsParameters object.")

        if isinstance(data_files, (str, Path)):
            self.control_obj._records_list.append(ControlRecord("data_file", [data_files], datatype=4))
        elif isinstance(data_files, list):
            self.control_obj._records_list.append(ControlRecord("data_file", data_files, datatype=4))
        else:
            raise ValueError("The supplied data files argument is invalid. Must be pathlike or list of paths.")

        self.control_obj._records_list.append(ControlRecord("model_output_file", [self.model_wd / 'prms.out'], datatype=4))

        if defaults:
            self.import_records_from_file(prep.root / prep.control_defaults)

    def import_records_from_file(self, fl: Union[str, Path], filetype: str = 'toml'):
        """
        Import PRMS control file variables from a toml or json file.
        Args:
            file: str | Path
                A file path to the input file.
            filetype: str
                String to tell which filetype parser to use, 'toml' - default, or 'json'

        Returns: None
            Appends the records to the ControlFile object.
        """
        if filetype == 'json':
            with open(fl, 'r') as jfile:
                file_dict = json.load(jfile)
        elif filetype == 'toml':
            with open(fl, 'rb') as tfile:
                file_dict = tomli.load(tfile)
        else:
            raise ValueError("The filetype argument is neither 'tomli' nor 'json'.")

        for key, item in file_dict.items():
            self.control_obj._records_list.append(ControlRecord(key, [item['value']], datatype=item['datatype']))

    # TODO - use prep.prms.meta to automatically assign variable dimensions from an input list of variables and then
    #   activate the proper output control parameters
    def select_output_variables(self, prms_vars: list):
        pass

    @staticmethod
    def load_from_prms_controlfile(cfile: Union[str, Path]):
        if isinstance(cfile, str):
            cfile = Path(str)

        gsflowcf = ControlFile.load_from_file(cfile)
        st = gsflowcf.get_record('start_time').values
        strt_time = f"{st.values[0]}-{st[1]}-{st[2]}"
        et = gsflowcf.get_record('end_time').values
        end_time = f"{et[0]}-{et[1]}-{et[2]}"
        datafiles = gsflowcf.get_record("data_file").values.tolist()
        paramfiles = gsflowcf.get_record("param_file").values.tolist()
        outpth = gsflowcf.get_record("model_output_file").values.tolist()
        modmode = gsflowcf.get_record("model_mode").values.tolist()

        return ControlBase(strt_time, end_time, datafiles, paramfiles, outpth, cfile.stem, modmode, defaults=False)


class StandardControl(ControlBase):

    def __init__(self,
                 start_time: str,
                 end_time: str,
                 data_files: Union[str, Path, list],
                 param_files: Union[str, Path, list, PrmsParameters],
                 output_path: Union[str, Path],
                 file_name: str = 'prms_model',
                 model_mode: str = 'PRMS5',
                 defaults: bool = True
                 ):

        super().__init__(start_time, end_time, data_files, param_files, output_path, file_name, model_mode, defaults)

        if defaults:
            self._select_default_vars()

    def _select_default_vars(self):
        with open(prep.root / prep.default_vars, 'rb') as vfile:
            varlists = tomli.load(vfile)

        bsnvars = varlists['basin']['vars']
        self.control_obj._records_list.append(ControlRecord("statsON_OFF", [1], datatype=1))
        self.control_obj._records_list.append(ControlRecord("nstatVars", [len(bsnvars)], datatype=1))
        self.control_obj._records_list.append(ControlRecord("statVar_element", ['1' for _ in bsnvars], datatype=4))
        self.control_obj._records_list.append(ControlRecord("stat_var_file", [self.model_wd / 'statvar.out'], datatype=4))
        self.control_obj._records_list.append(ControlRecord("statVar_names", bsnvars, datatype=4))


class HruOutControl(ControlBase):

    def __init__(self,
                 start_time: str,
                 end_time: str,
                 data_files: Union[str, Path, list],
                 param_files: Union[str, Path, list, PrmsParameters],
                 output_path: Union[str, Path],
                 file_name: str = 'prms_model',
                 model_mode: str = 'PRMS5',
                 defaults: bool = True
                 ):

        super().__init__(start_time, end_time, data_files, param_files, output_path, file_name, model_mode, defaults)

        if defaults:
            self._select_default_vars()

    def _select_default_vars(self):
        with open(prep.root / prep.default_vars, 'rb') as vfile:
            varlists = tomli.load(vfile)

        (self.model_wd / 'nhru_out').mkdir(parents=True, exist_ok=True)

        hruvars = varlists['hru']['vars']
        self.control_obj._records_list.append(ControlRecord("nhruOutON_OFF", [1], datatype=1))
        self.control_obj._records_list.append(ControlRecord("nhruOut_format", [2], datatype=1))
        self.control_obj._records_list.append(ControlRecord("nhruOut_freq", [1], datatype=1))
        self.control_obj._records_list.append(ControlRecord("nhruOutVars", [len(hruvars)], datatype=1))
        self.control_obj._records_list.append(ControlRecord("nhruOutBaseFileName", [self.model_wd / 'nhru_out/hru_'], datatype=4))
        self.control_obj._records_list.append(ControlRecord("nhruOutVar_names", hruvars, datatype=4))


class SubOutControl(ControlBase):

    def __init__(self,
                 start_time: str,
                 end_time: str,
                 data_files: Union[str, Path, list],
                 param_files: Union[str, Path, list, PrmsParameters],
                 output_path: Union[str, Path],
                 file_name: str = 'prms_model',
                 model_mode: str = 'PRMS5',
                 defaults: bool = True
                 ):

        super().__init__(start_time, end_time, data_files, param_files, output_path, file_name, model_mode, defaults)

        if defaults:
            self._select_default_vars()

    def _select_default_vars(self):
        with open(prep.root / prep.default_vars, 'rb') as vfile:
            varlists = tomli.load(vfile)

        (self.model_wd / 'nsub_out').mkdir(parents=True, exist_ok=True)

        subvars = varlists['subbasin']['vars']
        self.control_obj._records_list.append(ControlRecord("nsubOutON_OFF", [1], datatype=1))
        self.control_obj._records_list.append(ControlRecord("nsubOut_format", [2], datatype=1))
        self.control_obj._records_list.append(ControlRecord("nsubOut_freq", [1], datatype=1))
        self.control_obj._records_list.append(ControlRecord("nsubOutVars", [len(subvars)], datatype=1))
        self.control_obj._records_list.append(ControlRecord("nsubOutBaseFileName", [self.model_wd / 'nsub_out/subbasin_'], datatype=4))
        self.control_obj._records_list.append(ControlRecord("nsubOutVar_names", subvars, datatype=4))


class SegmentOutControl(ControlBase):

    def __init__(self,
                 start_time: str,
                 end_time: str,
                 data_files: Union[str, Path, list],
                 param_files: Union[str, Path, list, PrmsParameters],
                 output_path: Union[str, Path],
                 file_name: str = 'prms_model',
                 model_mode: str = 'PRMS5',
                 defaults: bool = True
                 ):

        super().__init__(start_time, end_time, data_files, param_files, output_path, file_name, model_mode, defaults)

        if defaults:
            self._select_default_vars()

    def _select_default_vars(self):
        with open(prep.root / prep.default_vars, 'rb') as vfile:
            varlists = tomli.load(vfile)

        (self.model_wd / 'nsegment_out').mkdir(parents=True, exist_ok=True)

        segvars = varlists['segment']['vars']
        self.control_obj._records_list.append(ControlRecord("nsegmentOutON_OFF", [1], datatype=1))
        self.control_obj._records_list.append(ControlRecord("nsegmentOut_format", [2], datatype=1))
        self.control_obj._records_list.append(ControlRecord("nsegmentOut_freq", [1], datatype=1))
        self.control_obj._records_list.append(ControlRecord("nsegmentOutVars", [len(segvars)], datatype=1))
        self.control_obj._records_list.append(ControlRecord("nsegmentOutBaseFileName", [self.model_wd / 'nsegment_out/segment_'], datatype=4))
        self.control_obj._records_list.append(ControlRecord("nsegmentOutVar_names", segvars, datatype=4))