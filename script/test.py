import os
import sys
import math
import pprint

import torch

from torchdrug import core
from torchdrug.utils import comm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from gnnqe import dataset, gnn, model, task, util

def test(cfg, solver):
    #solver.model.metric = ("mrr", "hits@1", "mape", "spearmanr")
    solver.model.metric = ("mrr", "hits@1", "hits@10", "mape", "spearmanr",
                           "Precision@0.3", "Precision@0.5", "Precision@0.7",
                          "Recall@0.3", "Recall@0.5", "Recall@0.7",
                          "Hard-Recall@0.3", "Hard-Recall@0.5", "Hard-Recall@0.7")
    #solver.evaluate("valid") only evaluate on test?
    solver.evaluate("test")


if __name__ == "__main__":
    torch.cuda.empty_cache()
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)

    torch.manual_seed(args.seed + comm.get_rank())

    logger = util.get_root_logger()
    if comm.get_rank() == 0:
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))

    dataset = core.Configurable.load_config_dict(cfg.dataset)
    solver = util.build_solver(cfg, dataset)

    #solver.load(cfg.checkpoint)

    test(cfg, solver)
