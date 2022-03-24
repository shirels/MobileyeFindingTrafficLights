from Model.model import TFLMan
import os
from pathlib import Path


"""
the controller:
input: play list
"""


def init(path: str):
    with open(path, 'r', encoding='utf-8') as pls:
        #play_list.pls
        absulote_path = path.replace('play_list.pls', '')
        text = pls.read()
        lines = text.split('\n')
        pkl = lines[0]
        pkl_path = os.path.join(absulote_path, Path(pkl).resolve())
        first_frame_index = int(lines[1])
        frames_path = lines[2:]
        frames_path = [os.path.join(absulote_path, Path(p).resolve()) for p in frames_path]
    tfl_manager = TFLMan(pkl_path)
    return pkl_path, frames_path, first_frame_index, tfl_manager


def run():
    pkl, frames_path, first_frame_index, tfl_manager = init(r'[FILL_DIRECTORY]\play_list.pls')
    for i, frame in enumerate(frames_path, start=first_frame_index):
        current_frame = frame
        prev_frame = None
        if i > first_frame_index:
            prev_frame = frames_path[i-25]
        tfl_manager.on_frame(i, current_frame, prev_frame, pkl)



if __name__ == "__main__":
    run()
