from pymol import cmd
import os
import pandas as pd


def init_worker():
    cmd.bg_color("white")
    cmd.hide("everything", "hydro")
    cmd.set("ray_opaque_background", 1)
    cmd.set("stick_ball", "on")
    cmd.set("stick_ball_ratio", 3.5)
    cmd.set("stick_radius", 0.15)
    cmd.set("sphere_scale", 0.2)
    cmd.set("valence", 1)
    cmd.set("valence_mode", 0)
    cmd.set("valence_size", 0.1)

    # any other one-time-per-process setup


def rotate_and_save(sdf_filepath: str, save_frames_path: str, sdf_index: str, sdf_features: pd.DataFrame, video_directory_path: str, sdf_list_len: int):
    
    print("evaluating sdf: ", sdf_index)
    print(sum(1 for entry in os.scandir(video_directory_path) if entry.is_dir()), "out of", sdf_list_len)
    print("core: ", os.getpid(), "\n")
    if not os.path.exists(save_frames_path) or (os.path.exists(save_frames_path) and sum(1 for entry in os.scandir(save_frames_path) if entry.is_file()) != 60):
        
        os.makedirs(save_frames_path, exist_ok=True)

        cmd.load(sdf_filepath)
        cmd.zoom("all", buffer=1)

        angle_size = 360 / 20
        count = 1
        for _, axis in enumerate(["x", "y", "z"]):
            for frame in range(1, 21):
                cmd.png(os.path.join(save_frames_path, f"{count}.png"))
                cmd.rotate(axis, angle_size)
                count += 1
        if sdf_index not in sdf_features["index"]:
            sdf_features = sdf_features[sdf_features["index"] != sdf_index]
        cmd.delete("mol")
    else:
        print("The frames for molecule with sdf index: " + sdf_index + " have already been generated in this directory")

