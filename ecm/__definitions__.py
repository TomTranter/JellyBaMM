# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 14:00:26 2020

@author: tom
"""

import ecm
import os

ROOT_DIR = os.path.dirname(os.path.abspath(ecm.__file__))
PARENT_DIR = os.path.dirname(ROOT_DIR)
TESTS_DIR = os.path.join(PARENT_DIR, 'tests')
FIXTURES_DIR = os.path.join(TESTS_DIR, 'fixtures')
TEST_CASES_DIR = os.path.join(FIXTURES_DIR, 'cases')
INPUT_DIR = os.path.join(ROOT_DIR, "input")
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
