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


Installation
^^^^^^^^^^^^

To install the latest stable version of CMB, you can use pip_ in a terminal:

.. code-block:: bash

    pip install -U cmb

For Martinos users, follow these guidelines:

1. Create new conda environment with correct version of Python:

.. code-block:: bash

    conda create --name (your-environment-name) python=3.8.6

2. clone git repo and switch to branch dev_nnunet:

.. code-block:: bash

    git clone https://github.com/johnsam7/ceremegbellum.git
    cd ceremegbellum
    git checkout dev_nnunet

3. install required public packages:

.. code-block:: bash

    pip install -r requirements.txt

4. install ceremegbellum package including antspy:

.. code-block:: bash

    python setup.py install

5. setup pre-compiled freesurfer (needed for segmentation):

.. code-block:: bash

    setfsvers
    setupfs

6. run example.py in examples folder. If connecting with remote desktop, mayavi will likely fail to render for graphics hardware reasons so then view needs to be set to 'flatmap' in plot_cerebellum_data() (which is the default). If at site, all viewing options should work.
7. Once you've confirmed that example.py runs until the end, you know the installation is successful and you can start using it for your own data. This is most easily done by modifying the example.py script by setting the paths to subject data to your own subject data.



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
