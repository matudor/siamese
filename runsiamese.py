#!/usr/bin/python
import sys, pylearn2, siamesenet
path=sys.argv[1]
from pylearn2.config import yaml_parse
tr=yaml_parse.load(open(path).read())
tr.main_loop()
