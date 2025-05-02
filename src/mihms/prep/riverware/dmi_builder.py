# This code is a constructor for a RiverWare Data Management Interface
# The DMI consists of an "external" executable that extracts selected time-series from the
# Model SQLite database and formats in RiverWare input file format to a standard directory in the same location
# as the database.

# The DMI class also creates the DMI control file that is read by RiverWare.

from pathlib import Path

from src.mihms.prep.riverware.rw_datafile import extract_series_from_db
from src.mihms.prep.riverware.rw_datafile import format_input_file


class StandardInputDMI:

    def __init__(self, input_slots, dbase_pth):

        self.in_slots = input_slots
        self.db_path = Path(dbase_pth)
        self.dmi_dir = Path(self.db_path.parent / "input_data")
        self.dmi_control_path = self.db_path

    def write_input_control(self):

        with open(self.db_path.parent / "input_dmi.control", 'w') as f:
            for d in self.in_slots:
                f.write('{0}: file="{1}" units={2} import={3}\n'.format(d['slot'],
                                                                      self.dmi_dir / d['slot'],
                                                                      d['units'],
                                                                      d['import']))

    def write_infiles(self, db_map):

        Path(self.dmi_dir).mkdir(parents=True, exist_ok=True)
        slot_lst = [val['slot'] for val in self.in_slots]
        for i in db_map:
            if i['slot'] in slot_lst:
                df = extract_series_from_db(self.db_path, i)
                s = {list(df.keys())[0]: df[list(df.keys())[0]][i['column']]}
                format_input_file(s, self.dmi_dir)
            else:
                continue

#class StandardOutputDMI:



if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
