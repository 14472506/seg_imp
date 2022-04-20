# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
from multiprocessing import Event
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.data import MetadataCatalog, build_detection_train_loader

from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.utils.events import EventStorage
from detectron2.evaluation import (
#    COCOEvaluator,
#    COCOPanopticEvaluator,
#    DatasetEvaluators,
#    LVISEvaluator,
#    PascalVOCDetectionEvaluator,
#    SemSegEvaluator,
     verify_results,
)

from adet.evaluation import Custom_COCOEvaluator

from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.utils.logger import setup_logger

from adet.data.dataset_mapper import DatasetMapperWithBasis
#from adet.data.fcpose_dataset_mapper import FCPoseDatasetMapper
from adet.config import get_cfg
from adet.checkpoint import AdetCheckpointer
#from adet.evaluation import TextEvaluator

from my_tools import custom_data_loader
from loss_eval_hook import LossEvalHook


class Trainer(DefaultTrainer):
    """
    This is the same Trainer except that we rewrite the
    `build_train_loader`/`resume_or_load` method.
    """
    def build_hooks(self):

        # calls hooks for training, these can either be modified or added to
        ret = super().build_hooks()
   
        # inserting loss eval hook into hook list
        ret.insert(-1, LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapperWithBasis(self.cfg, True)
            )
        ))
        
        # loop through and return all hooks
        for i in range(len(ret)):
            # Find the periodic Checkpoint hook in returned hooks list
            if isinstance(ret[i], hooks.PeriodicCheckpointer):

                # modify the defualt trainer checkpoint to add AdelaiDet checkpoint handler
                self.checkpointer = AdetCheckpointer(
                    self.model,
                    self.cfg.OUTPUT_DIR,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                )

                # add modification to hooks list
                ret[i] = hooks.PeriodicCheckpointer(self.checkpointer, self.cfg.SOLVER.CHECKPOINT_PERIOD)

            # return hooks with modifications
            return ret
    
    # Modify the resume_or_load function to use the AdelaiDet checkpointer
    def resume_or_load(self, resume=True):

        # define checkpoint
        checkpoint = self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        
        # define action to be taken in instance of resume and last checkpoint
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1

    # manages the training loop
    def train_loop(self, start_iter: int, max_iter: int):
        
        # gets log got adet training and log starting itteration infor in log
        logger = logging.getLogger("adet.trainer")
        logger.info("Starting training from iteration {}".format(start_iter))

        # set itter and start itter for this itteration then set max itterations
        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        # event storage is a detectron2 utility that handles data strage durin the training proces
        with EventStorage(start_iter) as self.storage:
            
            # handles tasks before storage
            self.before_train()                             

            # loops through iteration and handles tasks, before and after each step along with running step
            for self.iter in range(start_iter, max_iter):
                self.before_step()
                self.run_step()
                self.after_step()

            # function handles tasks after step
            self.after_train()

    # function is called to train the the model and calls the training loop within it
    def train(self):
        """
        Run training.
        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It calls :func:`detectron2.data.build_detection_train_loader` with a customized
        DatasetMapper, which adds categorical labels as a semantic mask.
        """

        #if cfg.MODEL.FCPOSE_ON:
        #    mapper = FCPoseDatasetMapper(cfg, True)
        #else:

        # mapper should be this, above removed because it should not be needed

        mapper = DatasetMapperWithBasis(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        
        # make output dir if it doesnt exist and add inference
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

        return Custom_COCOEvaluator(dataset_name, cfg, True, output_folder, use_fast_impl=False)

    # this whole thing may not be relevent to solov2
    #@classmethod
    #def test_with_TTA(cls, cfg, model):
    #    logger = logging.getLogger("adet.trainer")
    #    # In the end of training, run an evaluation with TTA
    #    # Only support some R-CNN models.
    #    logger.info("Running inference with test-time augmentation ...")
    #    model = GeneralizedRCNNWithTTA(cfg, model)
    #    evaluators = [
    #        cls.build_evaluator(
    #            cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
    #        )
    #        for name in cfg.DATASETS.TEST
    #    ]
    #    res = cls.test(cfg, model, evaluators)
    #    res = OrderedDict({k + "_TTA": v for k, v in res.items()})
    #    return res

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    rank = comm.get_rank()
    setup_logger(cfg.OUTPUT_DIR, distributed_rank=rank, name="adet")

    return cfg

def main(args):

    training_config_dict = {
        "train1": ["coco", "jr_train", "data/jersey_royal_ds/train/train.json", "data/jersey_royal_ds/train"],
    }
    testing_config_dict = {
        "test1": ["coco", "jr_val", "data/jersey_royal_ds/val/val.json", "data/jersey_royal_ds/val"],
        "test2": ["coco", "jr_val", "data/jersey_royal_ds/test/test.json", "data/jersey_royal_ds/test"]
    }
    thing_classes = ["Jersey Royal"]

    train_data = training_config_dict["train1"]
    training_meta = custom_data_loader(train_data[0], train_data[1], train_data[2], train_data[3], thing_classes)

    test_data = testing_config_dict["test2"]
    test_meta = custom_data_loader(test_data[0], test_data[1], test_data[2], test_data[3], thing_classes)

    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        AdetCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model) # d2 defaults.py
        if comm.is_main_process():
            verify_results(cfg, res)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )