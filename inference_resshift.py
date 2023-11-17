#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2023-03-11 17:17:41

import os, sys
import argparse
from pathlib import Path
import shutil  # Add this import for directory removal
import cv2
from omegaconf import OmegaConf
from sampler import ResShiftSampler
from utils.util_opts import str2bool
from basicsr.utils.download_util import load_file_from_url


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-i", "--in_path", type=str, default="", help="Input path.")
    parser.add_argument(
        "-o", "--out_path", type=str, default="./results", help="Output path."
    )
    parser.add_argument("-s", "--steps", type=int, default=15, help="Diffusion length.")
    parser.add_argument("--scale", type=int, default=4, help="Scale factor for SR.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed.")
    parser.add_argument(
        "--chop_size",
        type=int,
        default=512,
        choices=[512, 256],
        help="Chopping forward.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="realsrx4",
        choices=["realsrx4", "bicsrx4_opencv", "bicsrx4_matlab"],
        help="Chopping forward.",
    )
    args = parser.parse_args()

    return args


def get_configs(args):
    if args.task == "realsrx4":
        configs = OmegaConf.load("./configs/realsr_swinunet_realesrgan256.yaml")
    elif args.task == "bicsrx4_opencv":
        configs = OmegaConf.load("./configs/bicubic_swinunet_bicubic256.yaml")
    elif args.task == "bicsrx4_matlab":
        configs = OmegaConf.load("./configs/bicubic_swinunet_bicubic256.yaml")
        configs.diffusion.params.kappa = 2.0

    # prepare the checkpoint
    ckpt_dir = Path("./weights")
    if not ckpt_dir.exists():
        ckpt_dir.mkdir()
    ckpt_path = ckpt_dir / f"resshift_{args.task}_s{args.steps}.pth"
    if not ckpt_path.exists():
        load_file_from_url(
            url=f"https://github.com/zsyOAOA/ResShift/releases/download/v1.0/{ckpt_path.name}",
            model_dir=ckpt_dir,
            progress=True,
            file_name=ckpt_path.name,
        )
    vqgan_path = ckpt_dir / f"autoencoder_vq_f4.pth"
    if not vqgan_path.exists():
        load_file_from_url(
            url="https://github.com/zsyOAOA/ResShift/releases/download/v1.0/autoencoder_vq_f4.pth",
            model_dir=ckpt_dir,
            progress=True,
            file_name=vqgan_path.name,
        )

    configs.model.ckpt_path = str(ckpt_path)
    configs.diffusion.params.steps = args.steps
    configs.diffusion.params.sf = args.scale
    configs.autoencoder.ckpt_path = str(vqgan_path)

    # save folder
    if not Path(args.out_path).exists():
        Path(args.out_path).mkdir(parents=True)

    if args.chop_size == 512:
        chop_stride = 448
    elif args.chop_size == 256:
        chop_stride = 224
    else:
        raise ValueError("Chop size must be in [512, 384, 256]")

    return configs, chop_stride


def extract_frames(video_path, frames_folder, target_resolution):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Empty the output folder if it contains anything
    shutil.rmtree(frames_folder, ignore_errors=True)
    # Create the output folder if it doesn't exist
    os.makedirs(frames_folder, exist_ok=True)
    # Resize the frame before saving
    # Read and save each frame
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = resize_frame(frame, target_resolution)

        # Save the frame to the output folder
        frame_filename = f"frame_{frame_count:04d}.png"
        frame_path = os.path.join(frames_folder, frame_filename)
        cv2.imwrite(frame_path, frame)
        print("Screenshot saved")

        frame_count += 1

    # Release the video capture object
    cap.release()

    return fps, frames_folder


def create_video(input_folder, output_video_path, original_fps):
    frame_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".png")])

    # Read the first frame to get dimensions
    first_frame = cv2.imread(os.path.join(input_folder, frame_files[0]))
    height, width, _ = first_frame.shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )  # You can change the codec based on your needs
    out = cv2.VideoWriter(output_video_path, fourcc, original_fps, (width, height))

    # Write frames to the video
    for frame_file in frame_files:
        frame_path = os.path.join(input_folder, frame_file)
        frame = cv2.imread(frame_path)
        out.write(frame)

    # Release the VideoWriter and close all windows
    out.release()


def resize_frame(frame, target_resolution):
    # Resize the frame to the target resolution
    return cv2.resize(frame, target_resolution)


def resize_video(input_video_path, output_video_path, target_resolution):
    # Open the input video file
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # You can change the codec as needed
    out = cv2.VideoWriter(output_video_path, fourcc, fps, target_resolution)

    # Read and resize each frame, then write it to the output video
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame to the target resolution
        frame = resize_frame(frame, target_resolution)

        # Write the frame to the output video
        out.write(frame)

    # Release video capture and writer objects
    cap.release()
    out.release()


def main():
    args = get_parser()

    configs, chop_stride = get_configs(args)

    input_video_path = "media/ip_video/cat.mp4"
    frames_folder = "media/ip_frames"

    input_resolution = (640, 480)
    output_resolution = (1280, 720)

    fps, frames_folder = extract_frames(
        input_video_path, frames_folder, input_resolution
    )

    # After stitching frames, resize the final video to HD resolution

    resshift_sampler = ResShiftSampler(
        configs,
        chop_size=args.chop_size,
        chop_stride=chop_stride,
        chop_bs=1,
        use_fp16=True,
        seed=args.seed,
    )

    output_video_path = "media/op_video/op.mp4"
    output_video_path_HD = args.out_path
    SR_frames_folder = "media/hd_frames"

    resshift_sampler.inference(
        frames_folder, SR_frames_folder, bs=1, noise_repeat=False
    )
    create_video(SR_frames_folder, output_video_path, fps)

    resize_video(output_video_path, args.out_path, output_resolution)


if __name__ == "__main__":
    main()
