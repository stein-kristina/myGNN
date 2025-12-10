import logging
import sys
import torch.distributed as dist
from collections.abc import MutableMapping
from logging import getLogger

from ray import tune

from recbole.config import Config
from recbole.data import (
    create_dataset,
    data_preparation,
)
from recbole.data.transform import construct_transform
from recbole.utils import (
    init_logger,
    get_model,
    get_trainer,
    init_seed,
    set_color,
    get_flops,
    get_environment,
)

from models.mykgin_attn import KGIN

def run_recbole(
    model=None,
    dataset=None,
    config_file_list=None,
    config_dict=None,
    saved=True,
    queue=None,
):
    r"""A fast running api, which includes the complete process of
    training and testing a model on a specified dataset

    Args:
        model (str, optional): Model name. Defaults to ``None``.
        dataset (str, optional): Dataset name. Defaults to ``None``.
        config_file_list (list, optional): Config files used to modify experiment parameters. Defaults to ``None``.
        config_dict (dict, optional): Parameters dictionary used to modify experiment parameters. Defaults to ``None``.
        saved (bool, optional): Whether to save the model. Defaults to ``True``.
        queue (torch.multiprocessing.Queue, optional): The queue used to pass the result to the main process. Defaults to ``None``.
    """
    # configurations initialization
    config = Config(
        model=model,
        dataset=dataset,
        config_file_list=config_file_list,
        config_dict=config_dict,
    )
    # print(config) 已获取
    init_seed(config["seed"], config["reproducibility"])
    
    # tensorboard logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(sys.argv)
    logger.info(config)

    # dataset filtering
    print("create dataset...")
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    init_seed(config["seed"] + config["local_rank"], config["reproducibility"])
    # model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    model = KGIN(config, train_data._dataset).to(config["device"])
    logger.info(model)
    from recbole.model.knowledge_aware_recommender import CKE
    # print(model.__class__.__name__, model.__class__.__module__)
    # exit(0)
    
    # no transform
    transform = construct_transform(config)
    flops = get_flops(model, dataset, config["device"], logger, transform)
    logger.info(set_color("FLOPs", "blue") + f": {flops}")
    
    # trainer loading and initialization
    # from recbole.trainer import KGTrainer
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    # print(trainer.__class__.__name__)
    
    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=saved, show_progress=config["show_progress"]
    )

    # model evaluation
    test_result = trainer.evaluate(
        test_data, load_best_model=saved, show_progress=config["show_progress"]
    )

    environment_tb = get_environment(config)
    logger.info(
        "The running environment of this training is as follows:\n"
        + environment_tb.draw()
    )

    logger.info(set_color("best valid ", "yellow") + f": {best_valid_result}")
    logger.info(set_color("test result", "yellow") + f": {test_result}")

    result = {
        "best_valid_score": best_valid_score,
        "valid_score_bigger": config["valid_metric_bigger"],
        "best_valid_result": best_valid_result,
        "test_result": test_result,
    }

    if not config["single_spec"]:
        dist.destroy_process_group()

    if config["local_rank"] == 0 and queue is not None:
        queue.put(result)  # for multiprocessing, e.g., mp.spawn

    return result  # for the single process

# run_recbole(model='KGAT', dataset='ml-20m', config_file_list=["config/KGAT.yaml"])
# run_recbole(model='KGNNLS', dataset='ml-20m', config_file_list=["config/KGNNLS.yaml"])
run_recbole(model='KGIN', dataset='ml-1m', config_file_list=["config/KGIN.yaml"])

