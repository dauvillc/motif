"""Implements a Lightning module which receives several sources as input,
tokenizes them, masks a portion of the tokens, and trains a model to predict
the masked tokens."""

import torch

# Local imports
from motif.lightning_module.base_module import MultisourceAbstractModule
from motif.utils.visualization import display_realizations


class MultisourceAutoencoder(MultisourceAbstractModule):
    """Given a torch model which receives inputs from multiple sources, this class
    trains one autoencoder for a specific source type. This means that all sources
    should be of the same type and thus have the same dimensionality and same number
    of channels.
    The structure expects its input as a dict D {source_name: map}, where D[source] contains the
    following key-value pairs (all shapes excluding the batch dimension):
    - "id" is a list of strings of length (B,) each uniquely identifying the elements.
    - "avail" is a scalar tensor of shape (B,) containing 1 if the element is available
        and -1 otherwise.
    - "values" is a tensor of shape (B, C, ...) containing the values of the source.
    - "landmask" is a tensor of shape (B, ...) containing the land-sea mask of the source.
    - "coords" is a tensor of shape (B, 2, ...) containing the spatial coordinates of the source.
    - "dt" is a scalar tensor of shape (B,) containing the time delta between the reference time
        and the element's time, normalized by dt_max.
    - "dist_to_center" is a tensor of shape (B, ...) containing the distance
        to the center of the storm at each pixel, in km.

    The structure outputs a dict {source_name: tensor} containing the predicted values
    for each source.
    """

    def __init__(
        self,
        sources,
        cfg,
        backbone,
        adamw_kwargs,
        lr_scheduler_kwargs,
        loss_max_distance_from_center=None,
        ignore_land_pixels_in_loss=False,
        normalize_coords_across_sources=False,
        validation_dir=None,
        metrics={},
    ):
        """
        Args:
            sources (list of Source): Source objects defining the dataset's sources.
            cfg (dict): The whole configuration of the experiment. This will be saved
                within the checkpoints and can be used to rebuild the exact experiment.
            backbone (nn.Module): The backbone model to use for the autoencoder.
            adamw_kwargs (dict): Arguments to pass to torch.optim.AdamW (other than params).
            lr_scheduler_kwargs (dict): Arguments to pass to the learning rate scheduler.
            loss_max_distance_from_center (int or None): If specified, only pixels within this
                distance from the center of the storm (in km) will be considered
                in the loss computation.
            ignore_land_pixels_in_loss (bool): If True, the pixels that are on land
                will be ignored in the loss computation.
            normalize_coords_across_sources (bool): If True, the coordinates of each source
                will be normalized across all sources in the batch so that the minimum
                latitude and longitude in each example is 0 and maximum is 1.
                If False, the coordinates will be normalized as sinusoïds.
            validation_dir (optional, str or Path): Directory where to save the validation plots.
                If None, no plots will be saved.
            metrics (dict of str: callable): Metrics to compute during training and validation.
                A metric should have the signature metric(y_pred, batch, avail_flags, **kwargs)
                and return a dict {source: tensor of shape (batch_size,)}.
        """
        super().__init__(
            sources,
            cfg,
            adamw_kwargs=adamw_kwargs,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            loss_max_distance_from_center=loss_max_distance_from_center,
            ignore_land_pixels_in_loss=ignore_land_pixels_in_loss,
            normalize_coords_across_sources=normalize_coords_across_sources,
            validation_dir=validation_dir,
            metrics=metrics,
        )
        self.backbone = backbone

        # Check that all sources have the source type
        source_type = sources[0].type
        for source in sources:
            if source.type != source_type:
                raise ValueError(
                    f"All sources should have the same type, got {source.type} \
                                 and {source_type}"
                )

    def forward(self, x):
        """
        Args:
            x (dict of str -> dict of str -> tensor): The input batch
        Returns:
            dict of str -> tensor: The predicted values for each source.
        """
        pred = {}
        for source in x:
            pred[source] = self.backbone(x[source]["values"])
        return pred

    def compute_loss(self, pred, batch):
        # Availability flag: 1 if the token is available, -1 otherwise.
        avail_flags = {source: batch[source]["avail"] for source in batch}
        # The loss filtering function will only include samples whose flag is set to 0.
        # Here we want to consider all samples that are available (flag == 1).
        avail_flags = {source: (avail_flags[s] - 1) / 2 for s in avail_flags}

        # Filter the predictions and true values
        true_y = {source: batch[source]["values"] for source in batch}
        pred, true_y = super().apply_loss_mask(pred, true_y, batch, avail_flags)

        # Compute the loss
        losses = {}
        for source in pred:
            # Compute the loss for each source
            losses[source] = (pred[source] - true_y[source]).pow(2).mean()
        # If len(losses) == 0, i.e. for all masked sources the tokens were missing,
        # raise an error.
        if len(losses) == 0:
            raise ValueError("No tokens to compute the loss on")

        # Compute the total loss
        loss = sum(losses.values()) / len(losses)
        return loss

    def training_step(self, input_batch, batch_idx):
        batch_size = input_batch[list(input_batch.keys())[0]]["values"].shape[0]
        batch = self.preproc_input(input_batch)
        # Make predictions
        pred = self.forward(batch)
        # Compute the loss
        loss = self.compute_loss(pred, batch)

        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            on_step=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        return loss

    def validation_step(self, input_batch, batch_idx):
        batch = self.preproc_input(input_batch)
        # Make predictions
        pred = self.forward(batch)
        # Compute the loss
        loss = self.compute_loss(pred, batch)
        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            sync_dist=True,
        )

        avail_flags = {source: (batch[source]["avail"] - 1) / 2 for source in batch}
        if self.validation_dir is not None and batch_idx % 5 == 0:
            # For every 30 batches, make a prediction and display it.
            if batch_idx % 30 == 0:
                display_realizations(
                    pred,
                    input_batch,
                    avail_flags,
                    self.validation_dir / f"realizations_{batch_idx}",
                    deterministic=True,
                    display_fraction=0.25,
                )

        # Evaluate the metrics
        for metric_name, metric in self.metrics.items():
            metric_res = metric(pred, batch, avail_flags)
            # Compute the average metric over all sources
            avg_res = torch.stack(list(metric_res.values())).mean()
            self.log(
                f"val_{metric_name}",
                avg_res,
                on_epoch=True,
                on_step=False,
                sync_dist=True,
            )
        return loss

    def predict_step(self, batch, batch_idx):
        # TODO
        pass
