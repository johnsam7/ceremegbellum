# Example build:
#   docker build -t mne-tools/cmb:v0.01 .
#
# Example usage:
#   docker run -ti -v D:\Data\subjects:/workspace/subjects -v D:\Data\2_Processed\nnUNet:/workspace/nnUNet -v D:\Git\ceremegbellum:/workspace/ceremegbellum --name CMB mne-tools/cmb:v0.01
#   mri_convert -at /input/inputvolume.m3z /output/outvolume.mgz
#   recon-all -i /input/<t1_file.nii.gz> -subjid <subjectID> -all

# Start with nvidia pytorch configured ubuntu
FROM nvcr.io/nvidia/pytorch:21.08-py3

# ADD or Download FS_v6.0.0 from MGH and untar to /opt
ADD freesurfer-Linux-centos6_x86_64-stable-pub-v6.0.0.tar.gz /usr/local/
RUN apt-get update
RUN apt-get -y install bc binutils libgomp1 perl psmisc sudo tar tcsh unzip uuid-dev vim-common libjpeg62-dev libglu1-mesa libfreetype6 libxrender1 libfontconfig1
RUN mkdir /workspace/subjects
RUN mkdir /workspace/nnUNet

# Configure license 
COPY license.txt /usr/local/freesurfer/.license

ENV OS Linux
ENV FREESURFER_HOME /usr/local/freesurfer
ENV SUBJECTS_DIR /workspace/subjects

#nnUNet
ENV nnUNet_raw_data_base /workspace/nnUNet/nnUNet_raw
ENV nnUNet_preprocessed /workspace/nnUNet/nnUNet_preprocessed
ENV RESULTS_FOLDER /workspace/nnUNet/nnUNet_trained_models

# Configure bashrc to source FreeSurferEnv.sh
RUN /bin/bash -c ' echo -e "source $FREESURFER_HOME/FreeSurferEnv.sh &>/dev/null" >> /root/.bashrc '

# Configure CMB
RUN pip install cmb

# RUN pip install --upgrade .