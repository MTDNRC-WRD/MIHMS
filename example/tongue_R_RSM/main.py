# Run script for Tongue RiverWare example

from pathlib import Path

from mihms.models.model_config import RWConfig
from src.mihms.prep.riverware.dmi_builder import StandardInputDMI

config = RWConfig(Path('C:/Users/CNB968/OneDrive - MT/GitHub/MIHMS/example/tongue_R_RSM/tongue_rw_params.toml'))
dbpth = Path(config.model_info['database_path'])
in_slots = config.model_info['DMI_inputs']
dbmap = config.database_map['INPUTS']

dmi = StandardInputDMI(in_slots, dbpth)

dmi.write_input_control()

dmi.write_infiles(dbmap)




if __name__ == '__main__':
    config = Path('')
