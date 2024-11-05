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
import open3d as o3d
import cv2
from vrl3_agent import VRL3Agent
import utils
from termcolor import cprint, colored
from PIL import Image
import zarr
from copy import deepcopy
import numpy as np
import imageio
import pickle

from diffusion_policy_3d.gym_util.mjpc_wrapper import MujocoPointcloudWrapperAdroit


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='door', help='environment to run')
    parser.add_argument('--camera_name', type=str, default='top', help='camera name')
    parser.add_argument('--num_episodes', type=int, default=100, help='number of episodes to run')
    parser.add_argument('--root_dir', type=str, default='data', help='directory to save data')
    parser.add_argument('--expert_ckpt_path', type=str, default=None, help='path to expert ckpt')
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


def save_depth_image(image_array, save_path, quiet=False):
    """save image to file

    Args:
        image_array (ndarray): shape (H, W)
        save_path (str): path to save image
    """
    depth_map_normalized = ((image_array - image_array.min()) / (image_array.max() - image_array.min()) * 255).astype(np.uint8)
    image = Image.fromarray(depth_map_normalized)
    save_path = os.path.abspath(save_path)
    image.save(save_path)
    if not quiet:
        print(colored('[INFO]', 'blue'), f'Depth image saved to {save_path}')


def save_point_cloud_ply(array, file_path, quiet=False):
    """
    Save an N x 6 array as a PLY file.

    Parameters:
    array: numpy array of shape (N, 6)
    file_path: output filename with .ply extension
    """
    # Extract points and colors from the array
    points = array[:, 0: 3]  # x, y, z coordinates
    colors = array[:, 3: 6] / 255.0  # Normalize RGB colors to [0, 1]

    # Create Open3D point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # Write the point cloud to a PLY file
    o3d.io.write_point_cloud(file_path, point_cloud)

    if not quiet:
        print(colored('[INFO]', 'blue'), f'Point cloud saved to {file_path}')


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
    save_dir = os.path.join(args.root_dir, f'{args.env_name}_{args.camera_name}.zarr')
    video_dir = os.path.join(args.root_dir, 'visualized_data', 'videos', args.env_name, args.camera_name)
    image_dir = os.path.join(args.root_dir, 'visualized_data', 'images', args.env_name, args.camera_name)
    point_cloud_dir = os.path.join(args.root_dir, 'visualized_data', 'point_clouds', args.env_name, args.camera_name)
    depth_dir = os.path.join(args.root_dir, 'visualized_data', 'depths', args.env_name, args.camera_name)
    if os.path.exists(save_dir):
        cprint('Data already exists at {}'.format(save_dir), 'red')
        cprint("If you want to overwrite, delete the existing directory first.", "red")
        cprint("Do you want to overwrite? (y/n)", "red")
        # user_input = input()
        user_input = 'y'
        if user_input == 'y':
            cprint('Overwriting {}'.format(save_dir), 'red')
            os.system('rm -rf {}'.format(save_dir))
        else:
            cprint('Exiting', 'red')
            return
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(point_cloud_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    
    # load expert ckpt
    loaded_dict = torch.load(args.expert_ckpt_path, map_location='cpu')
    expert_agent = loaded_dict['agent']
    expert_agent.to('cuda')
    
    cprint('Loaded expert ckpt from {}'.format(args.expert_ckpt_path), 'green')

    total_count = 0
    img_arrays = []
    point_cloud_arrays = []
    depth_arrays = []
    state_arrays = []
    action_arrays = []
    episode_ends_arrays = []
    
    init_states = []

    # loop over episodes
    minimal_episode_length = 100
    episode_idx = 0
    while episode_idx < num_episodes:
        env = create_env()
        time_step = env.reset()
        input_obs_visual = time_step.observation # (3n,84,84), unit8
        input_obs_sensor = time_step.observation_sensor # float32, door(24,)q        

        init_state = env.env._env._env.get_env_state()

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
                # cam: top, vil_camera, fixed
                # vrl3_input = render_camera(env.env._env.sim, camera_name="top").transpose(2,0,1).copy() # (3,84,84)
                
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
            init_states.append(init_state)
            
            # ! Flip vertically: csx did this
            img_arrays_sub = [np.flip(image_array, axis=1) for image_array in img_arrays_sub]
            img_arrays_sub_np = np.stack(img_arrays_sub, axis=0).transpose(0,2,3,1)
            save_video_imageio(img_arrays_sub_np, os.path.join(video_dir, f'episode_{episode_idx}.mp4'), fps=30.0)
            save_rgb_image(img_arrays_sub_np[0], os.path.join(image_dir, f'episode_{episode_idx}_0.png'))
            save_depth_image(depth_arrays_sub[0], os.path.join(depth_dir, f'episode_{episode_idx}_0.png'))
            save_point_cloud_ply(point_cloud_arrays_sub[0], os.path.join(point_cloud_dir, f'episode_{episode_idx}_0.ply'))

            total_count += total_count_sub
            episode_ends_arrays.append(deepcopy(total_count)) # the index of the last step of the episode    
            img_arrays.extend(deepcopy(img_arrays_sub))
            point_cloud_arrays.extend(deepcopy(point_cloud_arrays_sub))
            depth_arrays.extend(deepcopy(depth_arrays_sub))
            state_arrays.extend(deepcopy(state_arrays_sub))
            action_arrays.extend(deepcopy(action_arrays_sub))
            print('Episode: {}, Reward: {}, Goal Achieved: {}'.format(episode_idx, total_reward, n_goal_achieved_total)) 
            episode_idx += 1

    # tracemalloc.stop()
    ###############################
    # save data
    ###############################
    pickle.dump(init_states, open(os.path.join(args.root_dir, f'{args.env_name}_{args.camera_name}_init_states.pkl'), 'wb'))
    # create zarr file
    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')
    # save img, state, action arrays into data, and episode ends arrays into meta
    img_arrays = np.stack(img_arrays, axis=0)
    if img_arrays.shape[1] == 3: # make channel last
        img_arrays = np.transpose(img_arrays, (0,2,3,1))
    state_arrays = np.stack(state_arrays, axis=0)
    point_cloud_arrays = np.stack(point_cloud_arrays, axis=0)
    depth_arrays = np.stack(depth_arrays, axis=0)
    action_arrays = np.stack(action_arrays, axis=0)
    episode_ends_arrays = np.array(episode_ends_arrays)

    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    img_chunk_size = (100, img_arrays.shape[1], img_arrays.shape[2], img_arrays.shape[3])
    state_chunk_size = (100, state_arrays.shape[1])
    point_cloud_chunk_size = (100, point_cloud_arrays.shape[1], point_cloud_arrays.shape[2])
    depth_chunk_size = (100, depth_arrays.shape[1], depth_arrays.shape[2])
    action_chunk_size = (100, action_arrays.shape[1])
    zarr_data.create_dataset('images', data=img_arrays, chunks=img_chunk_size, dtype='uint8', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('robot_states', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('point_clouds', data=point_cloud_arrays, chunks=point_cloud_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('depth', data=depth_arrays, chunks=depth_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('actions', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True, compressor=compressor)
    
    
    # print shape
    cprint(f'img shape: {img_arrays.shape}, range: [{np.min(img_arrays)}, {np.max(img_arrays)}]', 'green')
    cprint(f'point_cloud shape: {point_cloud_arrays.shape}, range: [{np.min(point_cloud_arrays)}, {np.max(point_cloud_arrays)}]', 'green')
    cprint(f'depth shape: {depth_arrays.shape}, range: [{np.min(depth_arrays)}, {np.max(depth_arrays)}]', 'green')
    cprint(f'state shape: {state_arrays.shape}, range: [{np.min(state_arrays)}, {np.max(state_arrays)}]', 'green')
    cprint(f'action shape: {action_arrays.shape}, range: [{np.min(action_arrays)}, {np.max(action_arrays)}]', 'green')
    cprint(f'Saved zarr file to {save_dir}', 'green')
    
    cprint(f'Saved zarr file to {save_dir}', 'green')
    
    # clean up
    del img_arrays, state_arrays, point_cloud_arrays, action_arrays, episode_ends_arrays
    del zarr_root, zarr_data, zarr_meta
    del env, expert_agent
    
    
if __name__ == '__main__':
    main()