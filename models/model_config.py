import os
import toml


class PRMSConfig:
    def __init__(self, config_path):
        self.elevation = None
        self.study_area_path = None
        try:
            config_data = toml.load(config_path)
        except FileNotFoundError:
            print(f'{config_path} not found')
            raise ValueError('No config file found!')

        model_info = config_data.get('MODEL_INFO', {})
        model_paths = config_data.get('MODEL_PATHS', {})
        input_paths = config_data.get('INPUT_PATHS', {})

        self.__dict__.update(model_info)
        self.__dict__.update(model_paths)
        self.__dict__.update(input_paths)

        # Convert 'units' values to integers
        for key in ['elev_units', 'precip_units', 'temp_units', 'runoff_units']:
            if key in self.__dict__:
                self.__dict__[key] = int(self.__dict__[key])

        # Split 'selected_stations' by comma and space
        if 'selected_stations' in self.__dict__:
            if self.__dict__['selected_stations']:
                self.__dict__['selected_stations'] = self.__dict__['selected_stations'].split(', ')
            else:
                self.__dict__['selected_stations'] = None

        # Prepend 'project_folder' to paths in 'INPUT_PATHS'
        project_folder = self.__dict__.get('project_folder', '')
        project_name = self.__dict__.get('project_name', '')
        hru_cellsize = self.__dict__.get('hru_cellsize', '')
        model_folder = '{}_{}'.format(project_name, str(hru_cellsize))

        for key, value in input_paths.items():
            self.__dict__[key] = os.path.join(project_folder, value)

        for key, value in model_paths.items():
            self.__dict__[key] = os.path.join(project_folder, model_folder, value)


if __name__ == '__main__':
    pass
# ========================= EOF ====================================================================
