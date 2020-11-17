# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 10:35:17 2020

@author: Tom
"""
import ecm
import configparser
import os

def load_config(path=None):
    if path is None:
        path = os.getcwd()
    config = configparser.ConfigParser()
    config.read(os.path.join(path, 'config.txt'))
    return config

def load_test_config():
    path = ecm.INPUT_DIR
    config = configparser.ConfigParser()
    config_fp = os.path.join(path, 'test_config.txt')
    config.read(config_fp)
    print('Config file', config_fp, 'loaded')
    print_config(config)
    return config

def print_config(config):
    for sec in config.sections():
        print('='*67)
        print(sec)
        print('='*67)
        for key in config[sec]:
            print('!', key.ljust(30, ' '), '!',
                  config.get(sec, key).ljust(30, ' '), '!')
            print('-'*67)
