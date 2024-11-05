# usage:
#       bash scripts/vrl3_gen_demonstration_expert.sh door
import mj_envs
from mjrl.utils.gym_env import GymEnv
from rrl_local.rrl_utils import make_basic_env, make_dir
from adroit import AdroitEnv
import matplotlib.pyplot as plt
import argparse
import os
import torch
import pickle
import open3d as o3d
import cv2
from vrl3_agent import VRL3Agent
import utils
from termcolor import cprint, colored
from PIL import Image
import zarr
import pathlib
from copy import deepcopy
import numpy as np
import imageio

from diffusion_policy_3d.gym_util.mjpc_wrapper import MujocoPointcloudWrapperAdroit


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='pen', help='environment to run')
    parser.add_argument('--camera_name', type=str, default='vil_camera', help='camera name')
    parser.add_argument('--num_episodes', type=int, default=1, help='number of episodes to run')
    parser.add_argument('--result_dir', type=str, default='/home/cvpr/Codes/EAI-Rrepresentation-Learning/results/sample_traj', help='directory to save data')
    parser.add_argument('--expert_ckpt_path', type=str, default='/home/cvpr/Codes/EAI-Rrepresentation-Learning/experts/vrl3_ckpts/vrl3_pen.pt', help='path to expert ckpt')
    parser.add_argument('--not_use_multi_view', action='store_true', help='not use multi view')
    parser.add_argument('--use_point_crop', action='store_true', help='use point crop')
    args = parser.parse_args()
    return args


def save_rgb_image(image_array, save_path, quiet=False):
    """save image to file

    Args:
        image_array (ndarray): shape (H, W, 3)
        save_path (str): path to save image
    """
    image = Image.fromarray(image_array)
    save_path = os.path.abspath(save_path)
    image.save(save_path)
    if not quiet:
        print(colored('[INFO]', 'blue'), f'RGB image saved to {save_path}')


def save_video_imageio(array, save_path, fps=30.0):
    """
    Save video from numpy array using imageio.
    Args:
        array (numpy.ndarray): (n_frames, height, width, 3)
        save_path (str): path to save video
        fps (float): frame per second
    """
    writer = imageio.get_writer(save_path, fps=fps)  # fps是帧率
    for frame in array:
        writer.append_data(frame)
    writer.close()
    print(f'Video (imageio) saved to {save_path}')


def main():
    args = parse_args()
    # load env
    action_repeat = 2
    frame_stack = 1
    def create_env():
        env = AdroitEnv(env_name=args.env_name+'-v0', test_image=False, num_repeats=action_repeat,
                        num_frames=frame_stack, env_feature_type='pixels',
                                            device='cuda', reward_rescale=True)
        env = MujocoPointcloudWrapperAdroit(env=env, env_name='adroit_'+args.env_name, use_point_crop=args.use_point_crop)
        return env
    num_episodes = args.num_episodes
    result_dir = pathlib.Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    # load expert ckpt
    loaded_dict = torch.load(args.expert_ckpt_path, map_location='cpu')
    expert_agent = loaded_dict['agent']
    expert_agent.to('cuda')
    cprint('Loaded expert ckpt from {}'.format(args.expert_ckpt_path), 'green')

    # loop over episodes
    minimal_episode_length = 100
    episode_idx = 0
    while episode_idx < num_episodes:
        env = create_env()
        time_step = env.reset()

        # ! save initial state
        init_state = env.env._env._env.get_env_state()
        pickle.dump(init_state, open(result_dir/'init_state.pkl', 'wb'))

        input_obs_visual = time_step.observation # (3n,84,84), unit8
        input_obs_sensor = time_step.observation_sensor # float32, door(24,)q        

        total_reward = 0.
        n_goal_achieved_total = 0.
        step_count = 0
        
        img_arrays_sub = []
        point_cloud_arrays_sub = []
        depth_arrays_sub = []
        state_arrays_sub = []
        action_arrays_sub = []
        total_count_sub = 0
        
        while (not time_step.last()) or step_count < minimal_episode_length:
            with torch.no_grad(), utils.eval_mode(expert_agent):
                input_obs_visual = time_step.observation
                input_obs_sensor = time_step.observation_sensor
                
                # ! csx did this
                action_image = input_obs_visual.transpose(1, 2, 0)
                action_image = cv2.resize(action_image, (84, 84), interpolation=cv2.INTER_AREA)
                action_image = action_image.transpose(2, 0, 1)

                action = expert_agent.act(obs=action_image, step=0,
                                        eval_mode=True, 
                                        obs_sensor=input_obs_sensor) # (28,) float32
                
                if args.not_use_multi_view:
                    input_obs_visual = input_obs_visual[:3] # (3,84,84)
                
                # save officail vil_camera view
                image = env.env._env._env.env.sim.render(width=224, height=224, mode='offscreen', camera_name='vil_camera_official', device_id=0)
                image = image.transpose(2, 0, 1) # (3,224,224)

                # save data
                total_count_sub += 1
                img_arrays_sub.append(image)
                state_arrays_sub.append(input_obs_sensor)
                action_arrays_sub.append(action)
                point_cloud_arrays_sub.append(time_step.observation_pointcloud)
                depth_arrays_sub.append(time_step.observation_depth)
                
            time_step = env.step(action)
            obs = time_step.observation # np array, (3,84,84)
            obs = obs[:3] if obs.shape[0] > 3 else obs # (3,84,84)
            n_goal_achieved_total += time_step.n_goal_achieved
            total_reward += time_step.reward
            step_count += 1
            
        if n_goal_achieved_total < 10.:
            cprint(f"Episode {episode_idx} has {n_goal_achieved_total} goals achieved and {total_reward} reward. Discarding.", 'red')
        else:
            # ! Flip vertically: csx did this
            img_arrays_sub = [np.flip(image_array, axis=1) for image_array in img_arrays_sub]
            img_arrays_sub_np = np.stack(img_arrays_sub, axis=0).transpose(0,2,3,1)
            save_video_imageio(img_arrays_sub_np, str(result_dir/'video.mp4'), fps=30.0)
            action_arrays_sub_np = np.stack(action_arrays_sub, axis=0)
            state_arrays_sub_np = np.stack(state_arrays_sub, axis=0)

            np.save(str(result_dir/'images.npy'), img_arrays_sub_np)
            np.save(str(result_dir/'actions.npy'), action_arrays_sub_np)
            np.save(str(result_dir/'states.npy'), state_arrays_sub_np)
            
            print('Episode: {}, Reward: {}, Goal Achieved: {}'.format(episode_idx, total_reward, n_goal_achieved_total)) 
            episode_idx += 1

    
    
if __name__ == '__main__':
    main()