from pathlib import Path
from typing import Union, Optional
from datetime import datetime
import json
import inspect

import tomli
from gsflow import ControlFile, ControlRecord
from gsflow import PrmsParameters
import numpy as np

from mihms.prep.prms import meta
from mihms.config import prep
from mihms.utils.io import user_warning, pathlike_as_string


class ControlBase:

    def __init__(self,
                 prms_exe: Union[str, Path],
                 prms_version: str,
                 start_time: str,
                 end_time: str,
                 data_files: Union[str, Path, list],
                 param_files: Union[str, Path, list, PrmsParameters],
                 output_path: Union[str, Path],
                 file_name: str = 'prms_model',
                 model_mode: str = 'PRMS5',
                 defaults: bool = True
                 ):

        self.control_obj = ControlFile(records_list=[], name=file_name)

        self.start_time = start_time
        self.end_time = end_time
        self.prms_exe = (prms_exe, prms_version)
        self._cfilenm = file_name
        self.model_wd = output_path
        self.param_file = param_files
        self.data_file = data_files
        self.model_mode = model_mode


        if defaults:
            self.import_records_from_file(prep.root / prep.control_defaults)

    @property
    def start_time(self):
        return self._start

    @start_time.setter
    def start_time(self, date: str):
        st = datetime.strptime(date, "%Y-%m-%d")
        self._start = date
        cntrl_st = [st.year, st.month, st.day, st.hour, st.minute, st.second]
        if 'start_time' in self.control_obj.record_names:
            self.control_obj.set_values('start_time', cntrl_st)
        else:
            self.control_obj._records_list.append(ControlRecord(
                "start_time",
                cntrl_st,
                datatype=1
            ))

    @property
    def end_time(self):
        return self._end

    @end_time.setter
    def end_time(self, date: str):
        en = datetime.strptime(date, "%Y-%m-%d")
        self._end = date
        cntrl_en = [en.year, en.month, en.day, en.hour, en.minute, en.second]
        if 'end_time' in self.control_obj.record_names:
            self.control_obj.set_values('end_time', cntrl_en)
        else:
            self.control_obj._records_list.append(ControlRecord(
                "end_time",
                cntrl_en,
                datatype=1
            ))

    @property
    def control_filename(self):
        return self._cfilenm

    @property
    def model_wd(self):
        return self._model_wd

    @model_wd.setter
    def model_wd(self, pth: Union[str, Path]):
        if isinstance(pth, str):
            self._model_wd = Path(pth)
        elif isinstance(pth, Path):
            self._model_wd = pth
        else:
            raise ValueError("The model output file path is not recognized as pathlike or does not exist.")

        self._model_wd.mkdir(parents=True, exist_ok=True)
        self._out_pth = self._model_wd / f"{Path(self.control_filename).stem}_output"
        self._out_pth.mkdir(parents=True, exist_ok=True)
        if 'model_output_file' in self.control_obj.record_names:
            self.control_obj.set_values("model_output_file", [(self._out_pth.relative_to(self._model_wd) / 'prms.out').as_posix()])
        else:
            self.control_obj._records_list.append(ControlRecord("model_output_file", [(self._out_pth.relative_to(self._model_wd) / 'prms.out').as_posix()], datatype=4))

        if 'csv_output_file' in self.control_obj.record_names:
            self.control_obj.set_values("csv_output_file", [(self._out_pth.relative_to(self._model_wd) / 'prms_summary.csv').as_posix()])
        else:
            self.control_obj._records_list.append(
                ControlRecord("csv_output_file", [(self._out_pth.relative_to(self._model_wd) / 'prms_summary.csv').as_posix()],
                              datatype=4))

    @property
    def prms_exe(self):
        return self._exepth

    @prms_exe.setter
    def prms_exe(self, exe_tup: tuple):
        if not isinstance(exe_tup, tuple):
            raise ValueError("Must provide a tuple of (prms.exe path, model version) to change the PRMS Executable path."
                             "The input argument is not of type tuple.")

        pth, ver = exe_tup
        pthstr = pathlike_as_string(pth)
        self._exepth = Path(pthstr)

        if 'executable_model' in self.control_obj.record_names:
            self.control_obj.set_values('executable_model', [pthstr])
        else:
            self.control_obj._records_list.append(ControlRecord('executable_model',
                                                                [pthstr],
                                                                datatype=4))

        if 'executable_desc' in self.control_obj.record_names:
            self.control_obj.set_values("executable_desc",[f"prms_{ver}"])
        else:
            self.control_obj._records_list.append(ControlRecord("executable_desc",
                                                                [f"prms_{ver}"],
                                                                datatype=4))

    @property
    def model_mode(self):
        return self._modmode

    @model_mode.setter
    def model_mode(self, modval: str):
        self._modmode = modval
        if 'model_mode' in self.control_obj.record_names:
            self.control_obj.set_values("model_mode", [self._modmode])
        else:
            self.control_obj._records_list.append(ControlRecord("model_mode", [self._modmode], datatype=4))

    @property
    def param_file(self):
        return self._paramfile

    @param_file.setter
    def param_file(self, new_pth: Union[str, Path, list, PrmsParameters]):
        if isinstance(new_pth, (str, Path)):
            new_pth = [new_pth]
        elif isinstance(new_pth, list):
            pass
        elif isinstance(new_pth, PrmsParameters):
            parameter_files = new_pth.parameter_files
            recl = []
            for f in parameter_files:
                if f is None:
                    recl.append(f"{Path(self.control_filename).stem}.param")
                else:
                    recl.append(f)
            new_pth = recl
        else:
            raise ValueError("The supplied parameter files argument is invalid. Must be pathlike, list of paths, or "
                             "pygsflow PrmsParameters object.")
        self._paramfile = new_pth
        if 'param_file' in self.control_obj.record_names:
            self.control_obj.set_values("param_file", pathlike_as_string(new_pth))
        else:
            self.control_obj._records_list.append(ControlRecord("param_file", pathlike_as_string(new_pth), datatype=4))

    @property
    def data_file(self):
        return self._datafile

    @data_file.setter
    def data_file(self, new_pth: Union[str, Path, list]):
        if isinstance(new_pth, (str, Path)):
            new_pth = [new_pth]
        elif isinstance(new_pth, list):
            pass
        else:
            raise ValueError("The supplied data files argument is invalid. Must be pathlike or list of paths.")

        self._datafile = new_pth
        if 'data_file' in self.control_obj.record_names:
            self.control_obj.set_values("data_file", pathlike_as_string(new_pth))
        else:
            self.control_obj._records_list.append(ControlRecord("data_file", pathlike_as_string(new_pth), datatype=4))

    def import_records_from_file(self, fl: Union[str, Path], filetype: str = 'toml'):
        """
        Import PRMS control file variables from a toml or json file.
        Args:
            fl: str | Path
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
            if key not in self.control_obj.record_names:
                if isinstance(item['value'], list):
                    self.control_obj._records_list.append(ControlRecord(key, item['value'], datatype=item['datatype']))
                else:
                    self.control_obj._records_list.append(ControlRecord(key, [item['value']], datatype=item['datatype']))
            else:
                if isinstance(item['value'], list):
                    self.control_obj.get_record(key).values = item['value']
                else:
                    self.control_obj.get_record(key).values = [item['value']]

    def add_statvars(self, vars: Union[str, list, np.ndarray], element_id: Union[int, list, np.ndarray]):
        """Adds additional output variables to the PRMS statvar output text file.

        Can supply a list of variables and the element id for each variable to include in the statvar
        output file. The statvar file can only display 1-dimension so for multi-dimension output like
        HRUs, segments, and sub-basins, you specify which HRU, segment, or sub-basin using its ID. If
        more than one element is desired for a given output variable it must be duplicated in the list
        of variable strings with the desired element index in the element_id list of integers.

        Args:
            vars: str | list | np.ndarray
                A string, list of strings, or array of strings representing valid PRMS output variables.
            element_id: int | list | np.ndarray
                An integer, list of integers, or array of integers representing the index of the specified
                model element (based on the variable).
                    - if using hru_ppt output variable entering element_id=1 would return HRU 1 precipitation

        Returns: None
            Appends the parameter settings to the control file.
        """
        if isinstance(vars, str):
            vars = [vars]

        if isinstance(vars, np.ndarray):
            vars = vars.tolist()

        if isinstance(element_id, int):
            element_id = [element_id]

        if isinstance(element_id, np.ndarray):
            element_id = element_id.tolist()

        if 'statsON_OFF' in self.control_obj.record_names:
            if self.control_obj.get_values('statsON_OFF')[0] == 0:
                self.control_obj.set_values('statsON_OFF', [1])
        else:
            self.control_obj._records_list.append(ControlRecord('statsON_OFF', [1], datatype=1))

        for v in vars:
            if not meta.is_available(v):
                raise ValueError(f"The requested output variable {v} is not available or invalid.")

        if 'statVar_names' in self.control_obj.record_names:
            self.control_obj.set_values('statVar_names',
                                        self.control_obj.get_values('statVar_names').tolist() + vars)
        else:
            self.control_obj._records_list.append(ControlRecord('statVar_names',
                                                                vars,
                                                                datatype=4))
        if 'statVar_element' in self.control_obj.record_names:
            self.control_obj.set_values(
                "statVar_element",
                self.control_obj.get_values('statVar_element').tolist() + [str(i) for i in element_id]
                )
        else:
            self.control_obj._records_list.append(ControlRecord(
                "statVar_element",
                [str(i) for i in element_id],
                datatype=4
                )
            )

        if 'nstatVars' in self.control_obj.record_names:
            self.control_obj.set_values("nstatVars",
                                        [len(self.control_obj.get_values('statVar_names'))])
        else:
            self.control_obj._records_list.append(ControlRecord("nstatVars",
                                                                [len(vars)],
                                                                datatype=1))

        if 'stat_var_file' in self.control_obj.record_names:
            pass
        else:
            self.control_obj._records_list.append(ControlRecord('stat_var_file',
            [(self._out_pth.relative_to(self.model_wd) / 'statvar.out').as_posix()],
                                                                datatype=4))

    def select_output_variables(self, prms_vars: list, overwrite: bool = True):
        """
        Loads a mixed list of variables and applies the relevant default settings for associated output dimension type.
        This only applies DEFAULT settings. Currently supported Dimensions are nhru, one, nsub, nsegment, nssr, ngw

        Args:
            prms_vars: list
                A list of prms output variables, can be hru, segment, subbasin, basin, or lake outputs. Settings for each
                output dimension type will be applied based on MIHMS defaults. If different settings for an output
                type are required it is easiest to just adjust the default value assigned to the ControlBase.control_obj
                pygsflow ControlFile object attribute.
            overwrite: bool
                Determines the behavior if variables are already listed for a given PRMS output module (e.g., if
                a basin variable already exists in the control file, overwrite=True will overwrite with the new set of
                basin variables included in prms_vars argument). If False, the prms_vars variables will be appended
                to the existing variable list in the control file. The default is True.

        Returns: None
            Applies default output settings and assigns output variables by dimension.

        """

        dimsvars = {}
        for v in prms_vars:
            vmeta = meta.find_variables(v)[v]
            dim = vmeta['dims'][0]
            if dim not in ['one', 'nsub', 'nhru', 'nssr', 'ngw', 'nsegment']:
                msg = f"Variable {v} with dimension {dim} is not currently supported, skipping record..."
                user_warning(
                    msg,
                    inspect.getframeinfo(
                        inspect.currentframe()
                    ),
                )
                continue
            if dim in dimsvars.keys():
                dimsvars[dim].append(v)
            else:
                dimsvars[dim] = [v]

        for k in dimsvars.keys():
            if k == 'one':
                (self._out_pth / 'basin_out').mkdir(parents=True, exist_ok=True)
                bsnvars = dimsvars[k]
                if 'basinOutVar_names' in self.control_obj.record_names:
                    if overwrite:
                        self.import_records_from_file(prep.root / prep.default_basinvars)
                        self.control_obj.get_record('basinOutVar_names').values = bsnvars
                        self.control_obj._records_list.append(ControlRecord("basinOutVars", [len(bsnvars)], datatype=1))
                        self.control_obj.get_record("basinOutBaseFileName").values = [
                            (self._out_pth.relative_to(
                                self.model_wd) / f"basin_out/{self.control_obj.get_record('basinOutBaseFileName').values[0]}").as_posix()]
                    else:
                        exst_vars = self.control_obj.get_values('basinOutVar_names').tolist()
                        self.import_records_from_file(prep.root / prep.default_basinvars)
                        self.control_obj.get_record('basinOutVar_names').values = exst_vars + bsnvars
                        self.control_obj._records_list.append(ControlRecord("basinOutVars", [len(exst_vars + bsnvars)], datatype=1))
                        self.control_obj.get_record("basinOutBaseFileName").values = [
                            (self._out_pth.relative_to(
                                self.model_wd) / f"basin_out/{self.control_obj.get_record('basinOutBaseFileName').values[0]}").as_posix()]
                else:
                    self.import_records_from_file(prep.root / prep.default_basinvars)
                    self.control_obj.get_record('basinOutVar_names').values = bsnvars
                    self.control_obj._records_list.append(ControlRecord("basinOutVars", [len(bsnvars)], datatype=1))
                    self.control_obj.get_record("basinOutBaseFileName").values = [
                        (self._out_pth.relative_to(
                            self.model_wd) / f"basin_out/{self.control_obj.get_record('basinOutBaseFileName').values[0]}").as_posix()]
            elif k == 'nsub':
                (self._out_pth / 'nsub_out').mkdir(parents=True, exist_ok=True)
                subvars = dimsvars[k]
                if 'nsubOutVar_names' in self.control_obj.record_names:
                    if overwrite:
                        self.import_records_from_file(prep.root / prep.default_subvars)
                        self.control_obj.get_record('nsubOutVar_names').values = subvars
                        self.control_obj._records_list.append(ControlRecord("nsubOutVars", [len(subvars)], datatype=1))
                        self.control_obj.get_record("nsubOutBaseFileName").values = [
                            (self._out_pth.relative_to(self.model_wd) / f"nsub_out/{self.control_obj.get_record('nsubOutBaseFileName').values[0]}").as_posix()]
                    else:
                        exst_vars = self.control_obj.get_values('nsubOutVar_names').tolist()
                        self.import_records_from_file(prep.root / prep.default_subvars)
                        self.control_obj.get_record('nsubOutVar_names').values = exst_vars + subvars
                        self.control_obj._records_list.append(ControlRecord("nsubOutVars", [len(exst_vars + subvars)], datatype=1))
                        self.control_obj.get_record("nsubOutBaseFileName").values = [
                            (self._out_pth.relative_to(
                                self.model_wd) / f"nsub_out/{self.control_obj.get_record('nsubOutBaseFileName').values[0]}").as_posix()]
                else:
                    self.import_records_from_file(prep.root / prep.default_subvars)
                    self.control_obj.get_record('nsubOutVar_names').values = subvars
                    self.control_obj._records_list.append(ControlRecord("nsubOutVars", [len(subvars)], datatype=1))
                    self.control_obj.get_record("nsubOutBaseFileName").values = [
                        (self._out_pth.relative_to(
                            self.model_wd) / f"nsub_out/{self.control_obj.get_record('nsubOutBaseFileName').values[0]}").as_posix()]
            elif k in ['nhru', 'nssr', 'ngw']:
                (self._out_pth / 'nhru_out').mkdir(parents=True, exist_ok=True)
                hruvars = dimsvars[k]
                if 'nhruOutVar_names' in self.control_obj.record_names:
                    if overwrite:
                        self.import_records_from_file(prep.root / prep.default_hruvars)
                        self.control_obj.get_record('nhruOutVar_names').values = hruvars
                        self.control_obj._records_list.append(ControlRecord("nhruOutVars", [len(hruvars)], datatype=1))
                        self.control_obj.get_record("nhruOutBaseFileName").values = [
                            (
                                        self._out_pth.relative_to(self.model_wd) / f"nhru_out/{self.control_obj.get_record('nhruOutBaseFileName').values[0]}").as_posix()]
                    else:
                        exst_vars = self.control_obj.get_values('nhruOutVar_names').tolist()
                        self.import_records_from_file(prep.root / prep.default_hruvars)
                        self.control_obj.get_record('nhruOutVar_names').values = exst_vars + hruvars
                        self.control_obj._records_list.append(ControlRecord("nhruOutVars", [len(exst_vars + hruvars)], datatype=1))
                        self.control_obj.get_record("nhruOutBaseFileName").values = [
                            (
                                    self._out_pth.relative_to(
                                        self.model_wd) / f"nhru_out/{self.control_obj.get_record('nhruOutBaseFileName').values[0]}").as_posix()]
                else:
                    self.import_records_from_file(prep.root / prep.default_hruvars)
                    self.control_obj.get_record('nhruOutVar_names').values = hruvars
                    self.control_obj._records_list.append(ControlRecord("nhruOutVars", [len(hruvars)], datatype=1))
                    self.control_obj.get_record("nhruOutBaseFileName").values = [
                        (
                                self._out_pth.relative_to(
                                    self.model_wd) / f"nhru_out/{self.control_obj.get_record('nhruOutBaseFileName').values[0]}").as_posix()]
            elif k == 'nsegment':
                (self._out_pth / 'nsegment_out').mkdir(parents=True, exist_ok=True)
                segvars = dimsvars[k]
                if 'nsegmentOutVar_names' in self.control_obj.record_names:
                    if overwrite:
                        self.import_records_from_file(prep.root / prep.default_segvars)
                        self.control_obj.get_record('nsegmentOutVar_names').values = segvars
                        self.control_obj._records_list.append(ControlRecord("nsegmentOutVars", [len(segvars)], datatype=1))
                        self.control_obj.get_record("nsegmentOutBaseFileName").values = [
                            (
                                        self._out_pth.relative_to(self.model_wd) / f"nsegment_out/{self.control_obj.get_record('nsegmentOutBaseFileName').values[0]}").as_posix()]
                    else:
                        exst_vars = self.control_obj.get_values('nsegmentOutVar_names').tolist()
                        self.import_records_from_file(prep.root / prep.default_segvars)
                        self.control_obj.get_record('nsegmentOutVar_names').values = exst_vars + segvars
                        self.control_obj._records_list.append(
                            ControlRecord("nsegmentOutVars", [len(exst_vars + segvars)], datatype=1))
                        self.control_obj.get_record("nsegmentOutBaseFileName").values = [
                            (
                                    self._out_pth.relative_to(
                                        self.model_wd) / f"nsegment_out/{self.control_obj.get_record('nsegmentOutBaseFileName').values[0]}").as_posix()]
                else:
                    self.import_records_from_file(prep.root / prep.default_segvars)
                    self.control_obj.get_record('nsegmentOutVar_names').values = segvars
                    self.control_obj._records_list.append(ControlRecord("nsegmentOutVars", [len(segvars)], datatype=1))
                    self.control_obj.get_record("nsegmentOutBaseFileName").values = [
                        (
                                self._out_pth.relative_to(
                                    self.model_wd) / f"nsegment_out/{self.control_obj.get_record('nsegmentOutBaseFileName').values[0]}").as_posix()]
            else:
                raise ValueError("The dimension is not recognized.")


    def save_controlfile(self):
        self.control_obj.write(self.model_wd / self.control_filename)

    @staticmethod
    def load_from_prms_controlfile(cfile: Union[str, Path]):
        if isinstance(cfile, str):
            cfile = Path(cfile)

        gsflowcf = ControlFile.load_from_file(cfile.as_posix())
        required_params = ['start_time', 'end_time', 'param_file', 'data_file', 'model_output_file', 'model_mode',
                           'executable_model', 'executable_desc']
        for i in required_params:
            if i not in gsflowcf.record_names:
                raise AttributeError(f"The required control parameter {i} is missing from the input control file.")
        mexe = gsflowcf.get_record('executable_model').values[0]
        exe_desc = gsflowcf.get_record('executable_desc').values[0]
        st = gsflowcf.get_record('start_time').values
        strt_time = f"{st[0]}-{st[1]}-{st[2]}"
        et = gsflowcf.get_record('end_time').values
        end_time = f"{et[0]}-{et[1]}-{et[2]}"
        datafiles = gsflowcf.get_record("data_file").values.tolist()
        paramfiles = gsflowcf.get_record("param_file").values.tolist()
        outpth = cfile.parent
        modmode = gsflowcf.get_record("model_mode").values[0]
        mihmcont = ControlBase(mexe, exe_desc, strt_time, end_time, datafiles, paramfiles, outpth, cfile.name, modmode,
                    defaults=False)
        for r in gsflowcf.records_list:
            if r.name in required_params:
                continue
            else:
                mihmcont.control_obj._records_list.append(r)

        return mihmcont


class CreateControlFile(ControlBase):
    def __init__(self,
                 prms_exe: Union[str, Path],
                 prms_version: str,
                 start_time: str,
                 end_time: str,
                 data_files: Union[str, Path, list],
                 param_files: Union[str, Path, list, PrmsParameters],
                 output_path: Union[str, Path],
                 file_name: str = 'prms_model.control',
                 model_mode: str = 'PRMS5',
                 defaults: bool = True
                 ):
        super().__init__(prms_exe, prms_version, start_time, end_time, data_files, param_files, output_path, file_name,
                         model_mode, defaults)


class StandardControl(ControlBase):

    def __init__(self,
                 prms_exe: Union[str, Path],
                 prms_version: str,
                 start_time: str,
                 end_time: str,
                 data_files: Union[str, Path, list],
                 param_files: Union[str, Path, list, PrmsParameters],
                 output_path: Union[str, Path],
                 file_name: str = 'prms_model.control',
                 model_mode: str = 'PRMS5',
                 defaults: bool = True
                 ):

        super().__init__(prms_exe, prms_version, start_time, end_time, data_files, param_files, output_path, file_name, model_mode, defaults)

        if defaults:
            self._select_default_vars()

    def _select_default_vars(self):
        self.import_records_from_file(prep.root / prep.default_statvars)

        bsnvars = self.control_obj.get_values('statVar_names')
        self.control_obj._records_list.append(ControlRecord("nstatVars", [len(bsnvars)], datatype=1))
        self.control_obj._records_list.append(ControlRecord("statVar_element", ['1' for _ in bsnvars], datatype=4))
        self.control_obj.get_record('stat_var_file').values = [(self._out_pth.relative_to(self.model_wd) / self.control_obj.get_record('stat_var_file').values[0]).as_posix()]


class HruOutControl(ControlBase):

    def __init__(self,
                 prms_exe: Union[str, Path],
                 prms_version: str,
                 start_time: str,
                 end_time: str,
                 data_files: Union[str, Path, list],
                 param_files: Union[str, Path, list, PrmsParameters],
                 output_path: Union[str, Path],
                 file_name: str = 'prms_model',
                 model_mode: str = 'PRMS5',
                 defaults: bool = True
                 ):

        super().__init__(prms_exe, prms_version, start_time, end_time, data_files, param_files, output_path, file_name, model_mode, defaults)

        if defaults:
            self._select_default_vars()

    def _select_default_vars(self):
        self.import_records_from_file(prep.root / prep.default_hruvars)

        (self._out_pth / 'nhru_out').mkdir(parents=True, exist_ok=True)

        hruvars = self.control_obj.get_values('nhruOutVar_names')
        self.control_obj._records_list.append(ControlRecord("nhruOutVars", [len(hruvars)], datatype=1))
        self.control_obj.get_record("nhruOutBaseFileName").values = [
            (self._out_pth.relative_to(self.model_wd) / f"nhru_out/{self.control_obj.get_record('nhruOutBaseFileName').values[0]}").as_posix()]


class SubOutControl(ControlBase):

    def __init__(self,
                 prms_exe: Union[str, Path],
                 prms_version: str,
                 start_time: str,
                 end_time: str,
                 data_files: Union[str, Path, list],
                 param_files: Union[str, Path, list, PrmsParameters],
                 output_path: Union[str, Path],
                 file_name: str = 'prms_model',
                 model_mode: str = 'PRMS5',
                 defaults: bool = True
                 ):

        super().__init__(prms_exe, prms_version, start_time, end_time, data_files, param_files, output_path, file_name, model_mode, defaults)

        if defaults:
            self._select_default_vars()

    def _select_default_vars(self):
        self.import_records_from_file(prep.root / prep.default_subvars)

        (self._out_pth / 'nsub_out').mkdir(parents=True, exist_ok=True)

        subvars = self.control_obj.get_values('nsubOutVar_names')
        self.control_obj._records_list.append(ControlRecord("nsubOutVars", [len(subvars)], datatype=1))
        self.control_obj.get_record("nsubOutBaseFileName").values = [
            (self._out_pth.relative_to(self.model_wd) / f"nsub_out/{self.control_obj.get_record('nsubOutBaseFileName').values[0]}").as_posix()]


class SegmentOutControl(ControlBase):

    def __init__(self,
                 prms_exe: Union[str, Path],
                 prms_version: str,
                 start_time: str,
                 end_time: str,
                 data_files: Union[str, Path, list],
                 param_files: Union[str, Path, list, PrmsParameters],
                 output_path: Union[str, Path],
                 file_name: str = 'prms_model',
                 model_mode: str = 'PRMS5',
                 defaults: bool = True
                 ):

        super().__init__(prms_exe, prms_version, start_time, end_time, data_files, param_files, output_path, file_name, model_mode, defaults)

        if defaults:
            self._select_default_vars()

    def _select_default_vars(self):
        self.import_records_from_file(prep.root / prep.default_segvars)

        (self._out_pth / 'nsegment_out').mkdir(parents=True, exist_ok=True)

        segvars = self.control_obj.get_values('nsegmentOutVar_names')
        self.control_obj._records_list.append(ControlRecord("nsegmentOutVars", [len(segvars)], datatype=1))
        self.control_obj.get_record("nsegmentOutBaseFileName").values = [
            (self._out_pth.relative_to(self.model_wd) / f"nsegment_out/{self.control_obj.get_record('nsegmentOutBaseFileName').values[0]}").as_posix()]