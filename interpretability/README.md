# Evaluating the interpretability metrics
In order to run evaluate the models according to the metrics used in the publication (localisation and pixel removal),
you may run 
```
python interpretability/analyses/localisation.py --save_path=MODEL_FOLDER --explainer_name=[Ours/GCam/...]
```
or
```
python interpretability/analyses/pixel_removal.py --save_path=MODEL_FOLDER --explainer_name=[Ours/GCam/...]
```

Configurations for the localisation experiment are included in `interpretability/analyses/localisation_configs.py`
and can be specified via the `--analysis_config` option. Further, the parameters for the explanation methods
can be updated by including an additional configuration in `interpretability/explanation_methods/explanation_configs.py`
and specified via the `--explainer_config` option. 
The model is expected to be named as `model_epoch_X.pkl` or `last_model_epoch_X.pkl` with `X` specifying the model epoch 
and saved in a path of the form `.../DATASET/BASENET/EXP_NAME` (e.g., `.../CIFAR10/final/9L-S-CoDA-SQ-1000`).

# Evaluating a model on video files
The script ``eval_on_video`` creates a video (see GIFs on the [main page](https://github.com/moboehle/CoDA-Nets#evaluated-on-videos)) in which the original video and 
attributions for `class_idx` are placed next to each other.
If no `class_idx` is given, the class that is predicted most often over all frames will be taken.
Further, `start_end` and `relative_box` can be used to evaluate on specific regions (`relative_box`) or 
on a contiguous subset of frames (`start_end`), both given in relative coordinates.  
E.g., the following line would evaluate the first 25% of the video `video.mp4` (placed in `VIDEO_PATH` in the configs of this folder)
for the class `97` (Mallard duck). For further info, see the file.

```
python interpretability/eval_on_video.py --video_file_name=video.mp4 --start_end=[0.0,0.25] --relative_box=[0.,1.,0.,1.] --class_idx=97
```
