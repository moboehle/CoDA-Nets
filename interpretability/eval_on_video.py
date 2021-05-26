import torch
import PIL
import argparse
import numpy as np
from importlib import import_module
from os.path import join

import cv2
from matplotlib import cm

from interpretability.configs import BASE_PATH, VIDEO_PATH, MAXV
from interpretability.utils import get_pretrained
from project_utils import Str2List, to_numpy


def argument_parser():
    """
    Create a parser with run_experiments arguments.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(description="Localisation metric analyser.")
    parser.add_argument("--save_path", default=join(BASE_PATH, "9L-L-CoDA-SQ-100000"),
                        help="Path for model checkpoints.")
    parser.add_argument("--model_config", default="9L-L-CoDA-SQ-100000",
                        type=str, help="Name of the model config. "
                                       "Should be a model included in the get_pretrained method.")
    parser.add_argument("--dataset", default="Imagenet",
                        type=str, help="Name of the dataset on which the model was trained. "
                                       "Should be a model included in the get_pretrained method.")
    parser.add_argument("--video_base_path", default=VIDEO_PATH,
                        type=str, help="Where to load videos from.")
    parser.add_argument("--video_file_name",
                        type=str, help="Video file name, e.g., 'video.mp4'.")
    parser.add_argument("--start_end", default=[0., 1.],
                        type=Str2List(dtype=float), help="Relative start end endpoint to eval in video.")
    parser.add_argument("--relative_box", default=[0., 1., 0., 1.],
                        type=Str2List(dtype=float), help="Relative box coordinates to extract video from."
                                                         "E.g., if only the upper half of the video "
                                                         "should be evaluated, this would be given as '0,1,0.5,1'.")
    parser.add_argument("--class_idx", default=-1,
                        type=int, help="Which class to show explanations for.")
    return parser


def get_arguments():
    parser = argument_parser()
    opts = parser.parse_args()
    return opts


def load_video(path, relative_box, relative_times):

    # Opens the Video file
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.cv2.CAP_PROP_FPS)
    frames = []
    w1, w2, h1, h2 = relative_box
    start, end = relative_times
    while (cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = frame.shape

        frames.append(frame[int(h1*h):int(h2*h), int(w1*w):int(w2*w)])

    cap.release()
    cv2.destroyAllWindows()
    total_frames = len(frames)

    return frames[int(start*total_frames):int(end*total_frames)], fps


@torch.no_grad()
def most_predicted(trainer, video):
    predictions = np.zeros(trainer.options["num_classes"])
    for img in video:
        img = trainer.data.get_test_loader().dataset.transform(PIL.Image.fromarray(img)).cuda()[None]
        predictions[trainer.predict(img).argmax()] += 1
    return np.argmax(predictions)


def att2img(attribution):
    return np.uint8(cm.bwr((np.clip(to_numpy(attribution) / MAXV, -1, 1) + 1) / 2) * 255)[:, :, :3]


def get_imgs_and_atts(trainer, video, class_idx):
    if class_idx == -1:
        class_idx = most_predicted(trainer, video)

    atts = []
    imgs = []
    for img in video:
        img = trainer.data.get_test_loader().dataset.transform(PIL.Image.fromarray(img)).cuda()[:][None]
        att = trainer.attribute(img, class_idx)[0].sum(0)
        atts.append(att2img(att))
        imgs.append(np.array(to_numpy(img[0].permute(1, 2, 0)) * 255, dtype=np.uint8))

    return imgs, atts


def save_video(imgs, atts, atts_path, fps):
    h, w = imgs[0].shape[:2]

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(atts_path, fourcc, fps, (2 * w, h))
    for img, att in zip(imgs, atts):
        _im = np.zeros((h, w * 2, 3), dtype=np.uint8)
        _im[:, :w] = img
        _im[:, w:] = att
        out.write(cv2.cvtColor(_im, cv2.COLOR_RGB2BGR))
    out.release()
    cv2.destroyAllWindows()


def main(config):
    trainer = get_pretrained(dataset=config.dataset, model=config.model_config)
    video_path = join(config.video_base_path, config.video_file_name)
    video, fps = load_video(video_path, config.relative_box, config.start_end)
    imgs, atts = get_imgs_and_atts(trainer, video, config.class_idx)
    suffix_start = video_path.rfind(".")
    atts_path = video_path[:suffix_start] + "_atts.mp4"
    save_video(imgs, atts, atts_path, fps)


if __name__ == "__main__":

    params = get_arguments()
    main(params)

