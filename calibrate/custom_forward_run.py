import os

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def run():
    p = '/home/dgketchum/PycharmProjects/MIHMS/run/run_mp.py'

    root = os.getcwd()
    control_file = 'control/smith_3000.control'
    parameter_file = 'input/smith_3000.params'
    data_file = 'input/smith_3000_xyz.data'

    python_path = '/home/dgketchum/PycharmProjects/MIHMS'
    os.environ['PYTHONPATH'] = python_path

    args = ['python' + ' {}'.format(p),
            '--project_dir', root,
            '--control', control_file,
            '--params', parameter_file,
            '--data', data_file,
            '--out_dir', os.path.join(os.getcwd(), 'pred')
            ]
    os.system(' '.join(args))


if __name__ == '__main__':
    run()

