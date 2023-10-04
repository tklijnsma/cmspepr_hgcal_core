#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup

with open("cmspepr_hgcal_core/include/VERSION", "r") as f:
    version = f.read().strip()

setup(
    name          = 'cmspepr_hgcal_core',
    version       = version,
    license       = 'BSD 3-Clause License',
    description   = 'Description text',
    url           = '',
    author        = 'Lindsey Gray <Lindsey.Gray@cern.ch>, Jan Kieseler <jan.kieseler@cern.ch>, Thomas Klijnsma <thomasklijnsma@gmail.com>',
    author_email  = 'thomasklijnsma@gmail.com',
    packages      = ['cmspepr_hgcal_core'],
    package_data  = {'cmspepr_hgcal_core': ['include/*']},
    include_package_data = True,
    )