import torch
import argparse
import os
import numpy as np

from interpretability.analyses.utils import load_trainer, Analyser
from interpretability.explanation_methods import get_explainer
from interpretability.explanation_methods.utils import limit_n_images
from project_utils import to_numpy
from interpretability.analyses.pixel_configs import configs


class PixelRemoveAnalyser(Analyser):

    def __init__(self, trainer, config_name, plotting_only=False, verbose=True, **config):
        """
        This analyser evaluates the pixel perturbation metric.
        Args:
            trainer: Trainer object.
            config_name: Configuration to load for the analysis itself. This config takes precedence over any
                passed parameter.
            plotting_only: Whether or not to load previous results.
            **config:
                explainer_config: Config key for the explanation configurations.
                explainer_name: Which explanation method to load. Default is Ours.

        """
        self.config_name = config_name
        analysis_config = configs[config_name]
        if verbose:
            for k in analysis_config:
                if k in config:
                    print("CAVE: Overwriting parameter:", k, analysis_config[k], config[k])
        config.update(analysis_config)
        super().__init__(trainer, **config)
        if plotting_only:
            self.load_results()
            return

        self.explainer = get_explainer(trainer, self.config["explainer_name"], self.config["explainer_config"])
        self.max_imgs_bs = 1

    def get_save_folder(self, epoch=None):
        """
        'Computes' the folder in which to store the results.
        Args:
            epoch: currently evaluated epoch.

        Returns: Path to save folder.

        """
        if epoch is None:
            epoch = self.trainer.epoch
        return os.path.join("pixel_removal_analysis", "epoch_{}".format(epoch),
                            self.config_name,
                            self.config["explainer_name"],
                            self.config["explainer_config"])

    def analysis(self):
        batch_size, n_imgs, sample_points, num_pixels = (self.config["batch_size"],
                                                         self.config["n_imgs"],
                                                         self.config["sample_points"],
                                                         self.config["num_pixels"])
        trainer = self.trainer
        loader = trainer.data.get_test_loader()
        results = []
        img_count = 0
        img_dims = None
        for img, tgt in loader:
            if img_dims is None:
                img_dims = np.prod(img.shape[-2:])
                max_idx = int(num_pixels * img_dims)
                stride = int(max_idx // sample_points)
            new_count = img_count + len(img)
            # Only evaluate n_imgs
            if new_count > n_imgs:
                img = img[:-(new_count - n_imgs)]
                tgt = tgt[:-(new_count - n_imgs)]
            img_count += len(img)
            # Evaluate the pixel perturbation for the sampled images
            results.append(self.process_imgs(img, tgt.argmax(1), batch_size, max_idx, stride))
            print("Done with {percent:5>.2f}%".format(percent=img_count * 100 / n_imgs), flush=True)
            if img_count == n_imgs:
                break
        return {"perturbation_results": to_numpy(torch.cat(results, dim=0).mean(0))}

    @limit_n_images
    def process_imgs(self, img, tgt, batch_size, num_pixels, stride):
        tgt = int(to_numpy(tgt))
        # Evaluate the given explainer on the image and sort the pixel indices by the 'importance values'
        pixel_importance = to_numpy(self.explainer.attribute(img.cuda(), target=tgt))[0].sum(0)
        idcs = np.argsort(pixel_importance.flatten())

        # Direction of the sorting
        if self.config["direction"] == "most":
            idcs = idcs[::-1]
        # Only delete the first num_pixels
        idcs = idcs[:num_pixels]

        # Compute the corresponding masks for deleting pixels in the given order
        positions = np.array(np.unravel_index(idcs, pixel_importance.shape)).T
        # First mask uses all pixels
        masks = [torch.ones(1, *pixel_importance.shape)]
        for h, w in positions:
            # Delete one additional position at a time
            mask = masks[-1].clone()
            mask[0, h, w] = 0
            masks.append(mask)

        # In order to speed up evaluation only evaluate masks at a stride (skipping the masks in between)
        masks = torch.cat([m for m_idx, m in enumerate(masks) if (m_idx % stride) == 0], dim=0)[:, None]
        # Compute the probabilities of the target class for the masked images
        # For efficiency, do this in batch mode.
        pert_out = torch.cat([self.masked_predict(img, masks[idx * batch_size: (idx + 1) * batch_size])
                              for idx in range(int(np.ceil(len(masks) / batch_size)))], dim=0)[None, :, tgt]
        return pert_out.cpu()

    @torch.no_grad()
    def masked_predict(self, img, masks):
        masked_imgs = self.trainer.pre_process_img(img.cuda()) * masks.cuda()
        return self.trainer.to_probabilities(self.trainer.model(masked_imgs))


def argument_parser():
    """
    Create a parser with run_experiments arguments.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(description="Localisation metric analyser.")
    parser.add_argument(
        "--save_path", default=None,
        help="Path for model checkpoints.")
    parser.add_argument("--batch_size", default=2,
                        type=int, help="Batch size for masked images.")
    parser.add_argument("--reload", default="last",
                        type=str, help="Which epoch to load. Options are 'last', 'best' and 'epoch_X',"
                                       "as long as epoch_X exists.")
    parser.add_argument("--explainer_name", default="Ours",
                        type=str, help="Which explainer method to use.")
    parser.add_argument("--explainer_config", default="test",
                        type=str, help="Which explainer configuration file to load.")
    parser.add_argument("--analysis_config", default="default",
                        type=str, help="Which analysis configuration file to load.")
    return parser


def get_arguments():
    parser = argument_parser()
    opts = parser.parse_args()
    return opts


def main(config):

    trainer = load_trainer(config.save_path, config.reload)

    analyser = PixelRemoveAnalyser(trainer, config.analysis_config, batch_size=config.batch_size,
                                   explainer_name=config.explainer_name, explainer_config=config.explainer_config)
    analyser.run()


if __name__ == "__main__":

    args = get_arguments()
    main(args)
