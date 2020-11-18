# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 11:24:19 2020

@author: tom
"""

import ecm


def test_func():
    test_config = ecm.load_test_config()
    ecm.print_config(test_config)


if __name__ == '__main__':
    test_func()
