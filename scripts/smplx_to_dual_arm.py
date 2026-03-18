"""
Fixed-base dual_arm retargeting from SMPL-X motion.

The base_link is welded to the world (no freejoint).
Only the 10 arm joints are solved by IK.

Usage:
    python scripts/smplx_to_dual_arm.py --smplx_file <path.npz> [--save_path <out.pkl>] [--loop] [--record_video]
"""

import argparse
import os
import pathlib
import pickle
import time

import mujoco
import mujoco.viewer as mjv
import numpy as np
from loop_rate_limiters import RateLimiter
from rich import print

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.params import ROBOT_XML_DICT
from general_motion_retargeting.utils.smpl import get_smplx_data_offline_fast, load_smplx_file

HERE = pathlib.Path(__file__).parent
ROBOT = "dual_arm"
SMPLX_FOLDER = HERE / ".." / "assets" / "body_models"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smplx_file", type=str,
                        default="ACCAD/Male2MartialArtsPunches_c3d/E3_-__cross_left_stageii.npz")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--video_path", type=str, default="videos/dual_arm_retarget.mp4")
    args = parser.parse_args()

    # ── Load SMPL-X ──────────────────────────────────────────────────────────
    smplx_data, body_model, smplx_output, human_height = load_smplx_file(
        args.smplx_file, SMPLX_FOLDER
    )
    smplx_frames, fps = get_smplx_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=30
    )
    print(f"[cyan]Loaded {len(smplx_frames)} frames @ {fps} fps, human height={human_height:.2f}m[/cyan]")

    # ── Init GMR retargeter ───────────────────────────────────────────────────
    retarget = GMR(
        actual_human_height=human_height,
        src_human="smplx",
        tgt_robot=ROBOT,
        verbose=False,
    )

    # ── Load MuJoCo model for viewer ─────────────────────────────────────────
    xml_path = str(ROBOT_XML_DICT[ROBOT])
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    n_joints = model.nq          # 10 hinge joints (no freejoint)
    rate = RateLimiter(frequency=fps, warn=False)

    # ── Optional video writer ─────────────────────────────────────────────────
    mp4_writer = None
    renderer = None
    if args.record_video:
        import imageio
        os.makedirs(os.path.dirname(args.video_path) or ".", exist_ok=True)
        mp4_writer = imageio.get_writer(args.video_path, fps=fps)
        renderer = mujoco.Renderer(model, height=480, width=640)
        print(f"[yellow]Recording to {args.video_path}[/yellow]")

    # ── Collect output if saving ──────────────────────────────────────────────
    qpos_list = [] if args.save_path is not None else None

    # ── Viewer loop ───────────────────────────────────────────────────────────
    with mjv.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        # Camera setup: front view of robot (arms face camera)
        viewer.cam.lookat = np.array([0.0, 0.1, 0.67])
        viewer.cam.distance = 1.3
        viewer.cam.elevation = -15
        viewer.cam.azimuth = 90

        i = 0
        fps_t0 = time.time()
        fps_count = 0

        print("[green]Viewer open. Press Ctrl+C or close window to stop.[/green]")

        while viewer.is_running():
            frame = smplx_frames[i]

            # Retarget: returns qpos of size n_joints (no root dof)
            qpos = retarget.retarget(frame)
            data.qpos[:] = qpos[:n_joints]
            mujoco.mj_forward(model, data)

            viewer.sync()
            rate.sleep()

            if qpos_list is not None:
                qpos_list.append(qpos[:n_joints].copy())

            if mp4_writer is not None:
                renderer.update_scene(data, camera=viewer.cam)
                mp4_writer.append_data(renderer.render())

            # FPS display
            fps_count += 1
            if time.time() - fps_t0 >= 2.0:
                print(f"FPS: {fps_count / (time.time() - fps_t0):.1f}")
                fps_count = 0
                fps_t0 = time.time()

            # Advance frame
            i += 1
            if i >= len(smplx_frames):
                if args.loop:
                    i = 0
                else:
                    break

    if mp4_writer is not None:
        mp4_writer.close()
        print(f"[green]Video saved to {args.video_path}[/green]")

    # ── Save motion data ──────────────────────────────────────────────────────
    if qpos_list is not None:
        dof_pos = np.array(qpos_list)          # (T, 10)
        motion_data = {
            "fps": fps,
            "dof_pos": dof_pos,                # joint angles j1-j10
            "joint_names": [f"j{i+1}" for i in range(n_joints)],
        }
        os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
        with open(args.save_path, "wb") as f:
            pickle.dump(motion_data, f)
        print(f"[green]Saved {dof_pos.shape[0]} frames to {args.save_path}[/green]")
        print(f"  dof_pos shape: {dof_pos.shape}")
        for k in range(n_joints):
            print(f"  j{k+1}: [{dof_pos[:,k].min():.3f}, {dof_pos[:,k].max():.3f}] rad")


if __name__ == "__main__":
    main()
