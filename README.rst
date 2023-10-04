.. -*- mode: rst -*-

|PyPI|_ |GH-CI|_

.. |PyPI| image:: https://badge.fury.io/py/cmb.svg?label=PyPI%20downloads
.. _PyPI: https://pypi.org/project/cmb/

.. |GH-CI| image:: https://github.com/johnsam7/ceremegbellum/actions/workflows/ci.yml/badge.svg?branch=main
.. _GH-CI: https://github.com/johnsam7/ceremegbellum/actions/workflows/ci.yml


Cere-MEG-Bellum (CMB) Package
=============================

Installation
^^^^^^^^^^^^

To install the latest stable version of CMB, you can use pip_ in a terminal:

.. code-block:: bash

    $ pip install -U cmb

For Martinos users
~~~~~~~~~~~~~~~~~~

1. Create new conda environment with correct version of Python:

.. code-block:: bash

    $ conda create --name <your-environment-name> python=3.8

2. clone git repo:

.. code-block:: bash

    $ git clone git@github.com:jasmainak/cbm.git
    $ cd cbm/

3. Activate the new environment:

.. code-block:: bash

    $ conda activate <your-environment-name>

4. install ceremegbellum package:

.. code-block:: bash

    $ pip install -e .

5. setup pre-compiled freesurfer (needed for segmentation):

.. code-block:: bash

    $ setfsvers
    $ setupfs

Running the example
^^^^^^^^^^^^^^^^^^^
1. Copy the cerebellum.geo file from /cluster/fusion/data/john_cerebellum/data
2. run example.py in examples folder. If connecting with remote desktop, mayavi will likely fail to render for graphics hardware reasons so then view needs to be set to 'flatmap' in plot_cerebellum_data() (which is the default). If at site, all viewing options should work.
3. Once you've confirmed that example.py runs until the end, you know the installation is successful and you can start using it for your own data. This is most easily done by modifying the example.py script by setting the paths to subject data to your own subject data.


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

Cite
^^^^

1. Samuelsson J G , Rosen B, Hamalainen M S. *Automatic Reconstruction of
Cerebellar Cortex from Standard MRI Using Diffeomorphic Registration of a
High-Resolution Template (ARCUS).* bioRxiv 2020.11.30.405522;
doi: https://doi.org/10.1101/2020.11.30.405522

.. _pip: https://pip.pypa.io/en/stable/
