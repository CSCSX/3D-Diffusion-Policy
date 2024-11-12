import pathlib
import subprocess
from termcolor import colored


TASKS = [
    # ('door', 30, 28),
    # ('hammer', 29, 26),
    ('pen', 24, 24),
]

CAMERAS = [
    'vil_camera',
]

SEED = [
    0,
]

cwd = pathlib.Path(__file__).resolve().parent/'3D-Diffusion-Policy'
code_root = pathlib.Path(__file__).resolve().parent.parent
tool_path = code_root/'3D-Diffusion-Policy'/'train_without_evaluation.py'

for seed in SEED:
    for task, state_dim, action_dim in TASKS:
        for camera in CAMERAS:
            cmd = [
                'python', str(tool_path),
                '--config-name=dp3_metaworld.yaml',
                f'task=adroit_no_runner',
                f'task.shape_meta.obs.agent_pos.shape=[{state_dim}]',
                f'task.shape_meta.action.shape=[{action_dim}]',
                f'hydra.run.dir=/data/Outputs/outputs_dp3/adroit_{task}_{camera}_seed{seed}',
                'training.debug=False',
                f'training.seed={seed}',
                'training.device=cuda:0',
                f'exp_name=adroit_{task}_{camera}_seed{seed}',
                f'logging.mode=offline',
                'checkpoint.save_ckpt=True',
                f'task.name={task}',
                f'task.dataset.zarr_path=/data/Data/DP3/Adroit/{task}_{camera}.zarr',
            ]
            print(colored('[INFO]', 'blue'), ' '.join(cmd))
            subprocess.run(cmd)
