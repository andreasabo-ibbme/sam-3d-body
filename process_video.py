# Copyright (c) Meta Platforms, Inc. and affiliates.
import argparse
import os, re
from glob import glob
import ffmpeg
import pyrootutils
import pickle, json

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", ".sl"],
    pythonpath=True,
    dotenv=True,
)

import cv2
import numpy as np
import torch
from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
from tools.vis_utils import (
    visualize_sample,
    visualize_sample_together,
    visualize_sample_2D,
    visualize_sample_2D_from_json,
    visualize_sample_3D_img,
    visualize_sample_3D_white,
)
from tqdm import tqdm
from icecream import ic

from load_json_utils import key_in_json, key_in_ref

process_local = True
if process_local:
    INPUT_FOLDER = r"/home/saboa/code/sam-3d-body/notebook/input_vids/BODY_JOINTS"
    # INPUT_FOLDER = r"/home/saboa/code/sam-3d-body/notebook/hi_res_input/BODY_JOINTS"
    BASE = r"/home/saboa/mnt/ndrive_andrea/AMBIENT/Andrea_S/EDS/SAM3D_DEC2025/dev"
    OUTPUT_FOLDER_SUBNAME = r"body_only_sam3_540x960_prompt_dev"

    # OUTPUT_FOLDER_VIDS = (
    #     r"/home/saboa/code/sam-3d-body/notebook/output_andrea_dev_bodyonly/BODY_JOINTS"
    # )
    # OUTPUT_FOLDER_PKL = r"/home/saboa/code/sam-3d-body/notebook/output_andrea_dev_body_only_pkl/BODY_JOINTS"

    OUTPUT_FOLDER_VIDS = os.path.join(
        BASE, "vids", OUTPUT_FOLDER_SUBNAME, "BODY_JOINTS"
    )
    OUTPUT_FOLDER_PKL = os.path.join(BASE, "pkl", OUTPUT_FOLDER_SUBNAME, "BODY_JOINTS")
else:

    INPUT_FOLDER = r"/home/saboa/mnt/ndrive_andrea/AMBIENT/Andrea_S/EDS/sorted_vids/formatted_input_downsampled_540x960/BODY_JOINTS"
    OUTPUT_FOLDER_VIDS = r"/home/saboa/mnt/ndrive_andrea/AMBIENT/Andrea_S/EDS/SAM3D_DEC2025/vids/formatted_input_downsampled_540x960/BODY_JOINTS"
    OUTPUT_FOLDER_PKL = r"/home/saboa/mnt/ndrive_andrea/AMBIENT/Andrea_S/EDS/SAM3D_DEC2025/pkl/formatted_input_downsampled_540x960/BODY_JOINTS"


# CHAH AI vids

INPUT_FOLDER = r"/home/saboa/mnt/ndrive_andrea/AMBIENT/Andrea_S/CHAH_AI/input_data/real"
OUTPUT_FOLDER_VIDS = r"/home/saboa/mnt/ndrive_andrea/AMBIENT/Andrea_S/CHAH_AI/output_data/dec_16/vids/real"
OUTPUT_FOLDER_PKL = r"/home/saboa/mnt/ndrive_andrea/AMBIENT/Andrea_S/CHAH_AI/output_data/dec_16/pkl/real"

INPUT_FOLDER = r"/home/asabo/scratch/sam-3d-body/notebook/images"
OUTPUT_FOLDER_VIDS = r"/home/asabo/scratch/sam-3d-body/notebook/output/vids"
OUTPUT_FOLDER_PKL = r"/home/asabo/scratch/sam-3d-body/notebook/output/pkl"


POSENET_KP_LOC = r"/home/saboa/mnt/ndrive_andrea/AMBIENT/Andrea_S/EDS/sorted_vids/pose_tracked_downsampled_540x960/BODY_JOINTS"

CHECKPOINT_PATH = (
    r"./checkpoints/sam-3d-body-dinov3/model.ckpt"
)
MHR_PATH = (
    r"./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt"
)
TEMP_IMG_FOLDER = r"./dev_chah"
DIGITS = 6
DETECTOR_NAME = "sam3"
INFERENCE_TYPE = "body"
PROMPT_KPS = False


def remake_folder(folder):
    if os.path.exists(folder):
        for f in os.listdir(folder):
            os.remove(os.path.join(folder, f))
    os.makedirs(folder, exist_ok=True)


def delete_folder(folder):
    for f in os.listdir(folder):
        os.remove(os.path.join(folder, f))


def purge(dir, pattern):
    for f in os.listdir(dir):
        if re.search(pattern, f):
            os.remove(os.path.join(dir, f))


def split_video(video_name, temp_img_folder):
    remake_folder(temp_img_folder)

    path, name = os.path.split(video_name)
    ffmpeg.input(video_name).filter("fps", fps=30).output(
        f"{temp_img_folder}/%0{DIGITS}d_Color.jpg"
    ).run(capture_stdout=True, capture_stderr=True)


def load_json_to_tensor(file_path):
    ref_size = 70
    try:
        with open(file_path, "r") as f:
            raw = json.load(f)

        # Sometimes frames are skipped, so identify the last frame
        last_frame = 0
        unique_frames = set()
        duplicate_frames = set()
        for el in raw:
            cur_frame = int(el)
            if cur_frame in unique_frames:
                duplicate_frames.add(cur_frame)
            else:
                unique_frames.add(cur_frame)
            last_frame = max(last_frame, cur_frame)

        last_frame += 1

        keypoints_prompt = torch.zeros((1, 1, 3, 1))
        keypoints_prompt[:, :, -1, :] = -2
        keypoints_prompt = keypoints_prompt.repeat(
            1, ref_size, 1, last_frame
        )  # [1, 70, 3, num_frames]
        ic(keypoints_prompt.shape)
        for ts in raw:
            for kp in key_in_json:
                # try:
                ts_int = int(ts)
                kp_ind_in_json = key_in_json[kp]
                kp_ind_in_out = key_in_ref[kp]

                # new_data = torch.tensor(raw[ts][0][kp_ind_in_json][0:2])

                keypoints_prompt[:, kp_ind_in_out, 0:2, ts_int] = torch.tensor(
                    raw[ts][0][kp_ind_in_json][0:2]
                ).to(keypoints_prompt)

                keypoints_prompt[0, kp_ind_in_out, 2, ts_int] = kp_ind_in_out
                # except Exception as e:
                # Didn't have data at this timestep
                # ic("laoding error", e)
                # quit()
                # pass
        return keypoints_prompt
    except Exception as e:
        ic(e)
        return None


def process_images(estimator, TEMP_IMG_FOLDER, OUTPUT_FOLDER, posenet_data):
    image_extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.gif",
        "*.bmp",
        "*.tiff",
        "*.webp",
    ]

    images_list = sorted(
        [
            image
            for ext in image_extensions
            for image in glob(os.path.join(TEMP_IMG_FOLDER, ext))
        ]
    )

    # Make all the folders we need once for the video
    OUTPUT_FOLDER_RGB = os.path.join(OUTPUT_FOLDER, "RGB_TEMP")
    OUTPUT_FOLDER_RGB_ORIG = os.path.join(OUTPUT_FOLDER, "RGB_ORIG_TEMP")
    OUTPUT_FOLDER_3D_IMG = os.path.join(OUTPUT_FOLDER, "3D_IMG_TEMP")

    remake_folder(OUTPUT_FOLDER_RGB)
    remake_folder(OUTPUT_FOLDER_RGB_ORIG)
    remake_folder(OUTPUT_FOLDER_3D_IMG)

    angles_to_render = [0, 90, 180, 270]
    for angle in angles_to_render:
        OUTPUT_FOLDER_ROT = os.path.join(OUTPUT_FOLDER, f"3D_{angle}_TEMP")
        remake_folder(OUTPUT_FOLDER_ROT)

    all_outputs = {}
    for image_path in tqdm(images_list):
        frame_num = int(os.path.split(image_path)[-1][:DIGITS])

        cur_ts_pose = None
        try:
            cur_ts_pose = posenet_data[..., frame_num - 1]
        except Exception as e:
            pass

        outputs = estimator.process_one_image(
            image_path,
            bbox_thr=args.bbox_thresh,
            use_mask=args.use_mask,
            inference_type=INFERENCE_TYPE,
            keypoints_from_2d=cur_ts_pose,
        )
        all_outputs[frame_num] = outputs
        img = cv2.imread(image_path)

        # if len(outputs) == 0:
        #     print(f"No humans detected in {os.path.basename(image_path)}, skipping...")
        #     continue

        # rend_img = visualize_sample_together(img, outputs, estimator.faces)
        rend_img_rgb = visualize_sample_2D(img, outputs, estimator.faces)
        rend_img_rgb_orig = visualize_sample_2D_from_json(
            img, outputs, estimator.faces, cur_ts_pose
        )

        cv2.imwrite(
            f"{OUTPUT_FOLDER_RGB}/{os.path.basename(image_path)[:-4]}.jpg",
            rend_img_rgb.astype(np.uint8),
        )
        cv2.imwrite(
            f"{OUTPUT_FOLDER_RGB_ORIG}/{os.path.basename(image_path)[:-4]}.jpg",
            rend_img_rgb_orig.astype(np.uint8),
        )

        rend_img_3d_img = visualize_sample_3D_img(img, outputs, estimator.faces)
        cv2.imwrite(
            f"{OUTPUT_FOLDER_3D_IMG}/{os.path.basename(image_path)[:-4]}.jpg",
            rend_img_3d_img.astype(np.uint8),
        )

        # Render different angles
        for angle in angles_to_render:
            OUTPUT_FOLDER_ROT = os.path.join(OUTPUT_FOLDER, f"3D_{angle}_TEMP")
            rend_img_3d_rot = visualize_sample_3D_white(
                img, outputs, estimator.faces, angle
            )
            cv2.imwrite(
                f"{OUTPUT_FOLDER_ROT}/{os.path.basename(image_path)[:-4]}.jpg",
                rend_img_3d_rot.astype(np.uint8),
            )

    return all_outputs


def make_vid_from_images(
    img_folder, OUTPUT_FOLDER, INPUT_FOLDER, video_name, ext=".avi"
):
    full_output_vid = video_name.replace(INPUT_FOLDER, OUTPUT_FOLDER)
    OUTPUT_FOLDER_PATH = os.path.split(full_output_vid)[0]
    video_name_no_ext = os.path.split(video_name)[-1][:-4]
    folder_prefix = os.path.split(img_folder)[-1][:-5]

    os.makedirs(OUTPUT_FOLDER_PATH, exist_ok=True)

    img_path = glob(os.path.join(img_folder, "*.jpg"))
    img_path = sorted(img_path)
    outfile = os.path.join(
        OUTPUT_FOLDER_PATH, f"{video_name_no_ext}_{folder_prefix}{ext}"
    )
    img = cv2.imread(img_path[0])
    height, width, _ = img.shape
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    if ext == ".mp4":
        fourcc = cv2.VideoWriter_fourcc(*"MP4V")

    writer = cv2.VideoWriter(outfile, fourcc, 30, (width, height))

    for z in range(0, len(img_path)):
        _php = img_path[z]

        if not os.path.isfile(_php):
            continue
        else:
            img = cv2.imread(_php)
            img = cv2.resize(img, (width, height))
            writer.write(img)
    writer.release()
    return outfile


def make_vids(OUTPUT_FOLDER, INPUT_FOLDER, video_name):
    # look for all temp folders
    temp_folders = glob(os.path.join(OUTPUT_FOLDER, "*_TEMP"))

    for folder in temp_folders:
        make_vid_from_images(folder, OUTPUT_FOLDER, INPUT_FOLDER, video_name)
        delete_folder(folder)


def process_single_video(video_name, estimator):
    ic("proccessing: ", video_name)

    if PROMPT_KPS:
        pose_track_file = video_name.replace(INPUT_FOLDER, POSENET_KP_LOC)
        pose_track_file = os.path.splitext(pose_track_file)[0]
        pose_track_file = os.path.join(pose_track_file, "posenet_data.json")
        posenet_data = load_json_to_tensor(pose_track_file)
    else:
        posenet_data = None
    split_video(video_name, TEMP_IMG_FOLDER)
    # TEMP_IMG_FOLDER = r"/home/saboa/code/sam-3d-body/notebook/temp_out_dev_small"

    all_output = process_images(
        estimator, TEMP_IMG_FOLDER, OUTPUT_FOLDER_VIDS, posenet_data
    )

    make_vids(OUTPUT_FOLDER_VIDS, INPUT_FOLDER, video_name)

    full_output_vid = video_name.replace(INPUT_FOLDER, OUTPUT_FOLDER_PKL)
    OUTPUT_FOLDER_PATH = os.path.split(full_output_vid)[0]
    os.makedirs(OUTPUT_FOLDER_PATH, exist_ok=True)
    video_name_no_ext = os.path.split(video_name)[-1][:-4]

    output_pkl = os.path.join(OUTPUT_FOLDER_PATH, f"{video_name_no_ext}.pkl")

    with open(output_pkl, "wb") as f:
        pickle.dump(all_output, f)


def main(args):

    os.makedirs(OUTPUT_FOLDER_VIDS, exist_ok=True)

    # Use command-line args or environment variables
    detector_path = args.detector_path or os.environ.get("SAM3D_DETECTOR_PATH", "")
    segmentor_path = args.segmentor_path or os.environ.get("SAM3D_SEGMENTOR_PATH", "")
    fov_path = args.fov_path or os.environ.get("SAM3D_FOV_PATH", "")

    # Initialize sam-3d-body model and other optional modules
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model, model_cfg = load_sam_3d_body(
        CHECKPOINT_PATH, device=device, mhr_path=MHR_PATH
    )

    human_detector, human_segmentor, fov_estimator = None, None, None
    if DETECTOR_NAME:
        from tools.build_detector import HumanDetector

        human_detector = HumanDetector(
            name=DETECTOR_NAME, device=device, path=detector_path
        )
    if len(segmentor_path):
        from tools.build_sam import HumanSegmentor

        human_segmentor = HumanSegmentor(
            name=args.segmentor_name, device=device, path=segmentor_path
        )
    if args.fov_name:
        from tools.build_fov_estimator import FOVEstimator

        fov_estimator = FOVEstimator(name=args.fov_name, device=device, path=fov_path)

    estimator = SAM3DBodyEstimator(
        sam_3d_body_model=model,
        model_cfg=model_cfg,
        human_detector=human_detector,
        human_segmentor=human_segmentor,
        fov_estimator=fov_estimator,
    )

    vid_extensions = ["*.mp4"]

    videos_list = sorted(
        [
            vid
            for ext in vid_extensions
            for vid in glob(os.path.join(INPUT_FOLDER, "**", ext), recursive=True)
        ]
    )
    ic(len(videos_list))

    for i, vid_name in enumerate(videos_list):
        if i % 10 == 0:
            ic(f"Processing {i} / {len(videos_list)} ")
        process_single_video(vid_name, estimator)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SAM 3D Body Demo - Single Image Human Mesh Recovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Examples:
                python demo.py --image_folder ./images --checkpoint_path ./checkpoints/model.ckpt

                Environment Variables:
                SAM3D_MHR_PATH: Path to MHR asset
                SAM3D_DETECTOR_PATH: Path to human detection model folder
                SAM3D_SEGMENTOR_PATH: Path to human segmentation model folder
                SAM3D_FOV_PATH: Path to fov estimation model folder
                """,
    )

    parser.add_argument(
        "--detector_name",
        default="vitdet",
        type=str,
        help="Human detection model for demo (Default `vitdet`, add your favorite detector if needed).",
    )
    parser.add_argument(
        "--segmentor_name",
        default="sam2",
        type=str,
        help="Human segmentation model for demo (Default `sam2`, add your favorite segmentor if needed).",
    )
    parser.add_argument(
        "--fov_name",
        default="moge2",
        type=str,
        help="FOV estimation model for demo (Default `moge2`, add your favorite fov estimator if needed).",
    )
    parser.add_argument(
        "--detector_path",
        default="",
        type=str,
        help="Path to human detection model folder (or set SAM3D_DETECTOR_PATH)",
    )
    parser.add_argument(
        "--segmentor_path",
        default="",
        type=str,
        help="Path to human segmentation model folder (or set SAM3D_SEGMENTOR_PATH)",
    )
    parser.add_argument(
        "--fov_path",
        default="",
        type=str,
        help="Path to fov estimation model folder (or set SAM3D_FOV_PATH)",
    )

    parser.add_argument(
        "--bbox_thresh",
        default=0.8,
        type=float,
        help="Bounding box detection threshold",
    )
    parser.add_argument(
        "--use_mask",
        action="store_true",
        default=True,
        help="Use mask-conditioned prediction (segmentation mask is automatically generated from bbox)",
    )
    args = parser.parse_args()

    main(args)
