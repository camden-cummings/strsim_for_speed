import pickle

import cv2
import numpy as np
from scipy.stats import mode


def calc_mode(deq: np.ndarray, frame_height: int, frame_width: int):
    """Finds a mode image based on a given array of images, effectively creating a background
    image.
    """
    
    mode_img, _ = np.array(mode(deq, axis=0, keepdims=False), dtype=np.uint8)
    
    return mode_img


def calc_mode_img(vidcap, frame_width: int, frame_height: int, filename: str, blur: bool, sections=50):
    """Calculates mode image based on all images in video, creating a version of the image with."""
    moviedeq = []
    cont, frame = vidcap.read()
    curr_frame = 0

    while cont:
        curr_frame += int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT) / sections)

        vidcap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)

        cont, frame = vidcap.read()

        if not cont:
            break

        if blur:
            stored_frame = cv2.GaussianBlur(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (7, 7), 0)
        else:
            stored_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        moviedeq.append(stored_frame)

    mode_img = calc_mode(moviedeq, frame_height, frame_width)
    cv2.imwrite(filename, mode_img)

    _ = vidcap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    return mode_img

def get_contour_mask(cell_filename, frame_width, frame_height):
    """
    Parameters
    ----------
    cell_filename
    frame_width
    frame_height

    Returns
    -------
    unknown
    """
    contour_mask = np.zeros((frame_height, frame_width, 3))

    with open(cell_filename, 'rb') as f:
        rois = pickle.load(f)

        for c in rois:
            contour_mask = cv2.drawContours(contour_mask, [np.array(c, dtype=np.int64)],
                                            -1, (255, 255, 255), thickness=cv2.FILLED)

    contour_mask = cv2.cvtColor(
        np.array(contour_mask, dtype=np.uint8), cv2.COLOR_BGR2GRAY)

    return contour_mask