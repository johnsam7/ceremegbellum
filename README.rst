.. -*- mode: rst -*-

|PyPI|_ |GH-CI|_

.. |PyPI| image:: https://badge.fury.io/py/cmb.svg?label=PyPI%20downloads
.. _PyPI: https://pypi.org/project/cmb/

.. |GH-CI| image:: https://github.com/johnsam7/ceremegbellum/actions/workflows/ci.yml/badge.svg?branch=main
.. _GH-CI: https://github.com/johnsam7/ceremegbellum/actions/workflows/ci.yml


Cere-MEG-Bellum (CMB) Package
=============================

For more information about CMB, please read the following paper:

  Samuelsson J G , Rosen B, Hamalainen M S. *Automatic Reconstruction of Cerebellar Cortex from Standard MRI Using Diffeomorphic Registration of a High-Resolution Template (ARCUS).* bioRxiv 2020.11.30.405522; doi: https://doi.org/10.1101/2020.11.30.405522

To use this repo, you will need the cerebellar atlas data which are available here; https://osf.io/98p3a/?view_only=933654b10152444992b9e7d8ff9f1112


Installation
^^^^^^^^^^^^

To install the latest stable version of CMB, you can use pip_ in a terminal:

.. code-block:: bash

    pip install -U cmb


Usage of the Docker Container
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

before building the image please place in this folder the freesurfer archive 'freesurfer-Linux-centos6_x86_64-stable-pub-v6.0.0.tar.gz' along with the freesurfer license file 'license.txt'.

Subsequently build the docker image with

.. code-block:: bash

    docker build -t mne-tools/cmb:v0.01 .

and run it with

.. code-block:: bash

    docker run -ti -v <YOUR SUBJECTS DIR>:/workspace/subjects -v <YOUR PROCESSED nnUNet DIR>:/workspace/nnUNet -v <YOUR ceremegbellum GIT DIR>:/workspace/ceremegbellum --name CMB mne-tools/cmb:v0.01

It is convinient to install CMB for development directly from the local repository. Change the directory to '/workspace/ceremegbellum' in the CLI of the Docker Container and run

.. code-block:: bash

    pip install -e .


Licensing
^^^^^^^^^
CMB is **MIT-licensed**:

    Copyright (c) 2021-2022, authors of CMB.
    All rights reserved.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    **THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.**


.. _pip: https://pip.pypa.io/en/stable/
