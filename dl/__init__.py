#!/usr/bin/env
# -*- coding: utf-8 -*-

"""
__init__.py.py: 
"""
import os.path as osp
import json
from dl.config import get_arguments
import logging
from pathlib import Path


def get_logger(args):
    formatter = logging.Formatter('%(asctime)s %(levelname)s - %(funcName)s(%(lineno)d): %(message)s',
                                  "%H:%M:%S")
    logger = logging.getLogger(__name__)
    logger.setLevel(args.log_level.upper())
    stream = logging.StreamHandler()
    stream.setLevel(args.log_level.upper())
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    return logger


args = get_arguments()
logger = get_logger(args)
