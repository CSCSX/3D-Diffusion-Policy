import pathlib
import subprocess
from termcolor import colored


TASKS = [
    # 'assembly',
    'button-press',
    'bin-picking',
    'hammer',
    'drawer-open',
    'push-wall',
    'reach',
    'shelf-place',
    'sweep-into',
    'hand-insert',
    'handle-pull',
    'box-close',
    'peg-unplug-side',
    'dial-turn',
    'lever-pull',
]

CAMERAS = [
    'corner',
    'corner2',
]

SEED = [
    0,
]

cwd = pathlib.Path(__file__).resolve().parent/'3D-Diffusion-Policy'
code_root = pathlib.Path(__file__).resolve().parent.parent
tool_path = code_root/'3D-Diffusion-Policy'/'train_without_evaluation.py'

for seed in SEED:
    for task in TASKS:
        for camera in CAMERAS:
            cmd = [
                'python', str(tool_path),
                '--config-name=dp3_metaworld.yaml',
                f'task=metaworld_no_runner',
                f'hydra.run.dir=/home/cscsx/Codes/EAI-Rrepresentation-Learning/outputs_dp3/metaworld_{task}_{camera}_seed{seed}',
                'training.debug=False',
                f'training.seed={seed}',
                'training.device=cuda:0',
                f'exp_name=metaworld_{task}_{camera}_seed{seed}',
                f'logging.mode=offline',
                'checkpoint.save_ckpt=True',
                f'task.name={task}',
                f'task.dataset.zarr_path=/data/Data/DP3/Metaworld/metaworld_{task}_{camera}_expert.zarr',
            ]
            print(colored('[INFO]', 'blue'), ' '.join(cmd))
            subprocess.run(cmd)
