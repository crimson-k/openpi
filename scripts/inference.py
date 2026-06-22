import dataclasses

import jax
import importlib
import argparse

from openpi.models import model as _model
from openpi.policies import alhoa_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

from 

from script.eval_policy import parse_args_and_config

def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance

def main(usr_args):
    task_name = usr_args["task_name"]

    TASK_ENV = class_decorator(usr_args["task_name"])




if __name__ == "__main__":

    usr_args = parse_args_and_config()

    main(usr_args)
