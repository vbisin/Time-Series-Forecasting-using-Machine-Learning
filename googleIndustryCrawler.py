#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 16:54:42 2016

@author: Vitto
"""
from urllib import urlopen
from lxml.html import parse

def industry(name):
  tree = parse(urlopen('http://www.google.com/finance?&q='+name))
  return tree.xpath("//a[@id='sector']")[0].text, tree.xpath("//a[@id='sector']")[0].getnext().text
