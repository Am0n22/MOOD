FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-devel
# copy files
ADD scripts /workspace/
RUN chmod +x /workspace/*.sh
RUN mkdir /mnt/data
RUN mkdir /mnt/pred
RUN pip install nibabel
RUN pip install h5py
