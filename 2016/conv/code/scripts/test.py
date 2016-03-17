# -*- coding: utf-8 -*-
"""
Created on Sun May 24 15:10:03 2015

@author: zjh
"""
import os
for parent,dirnames,filenames in os.walk('F:/test'):
    print type(parent)