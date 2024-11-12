# bash scripts/gen_demonstration_adroit.sh door
# bash scripts/gen_demonstration_adroit.sh hammer
# bash scripts/gen_demonstration_adroit.sh pen

cd third_party/VRL3/src

task=${1}

CUDA_VISIBLE_DEVICES=0 python gen_demonstration_expert.py --env_name $task \
                        --num_episodes 10 \
                        --root_dir "../../../3D-Diffusion-Policy/data/" \
                        --expert_ckpt_path "../ckpts/vrl3_${task}.pt" \
                        --img_size 84 \
                        --not_use_multi_view \
                        --use_point_crop


python gen_demonstration_expert.py --env_name pen \
    --num_episodes 120 \
    --root_dir "/data/Data/DP3_RAW" \
    --expert_ckpt_path "/home/cscsx/Codes/EAI-RL-CSX/experts/vrl3_ckpts/vrl3_pen.pt" \
    --not_use_multi_view \
    --camera_name vil_camera \
    --use_point_crop
    