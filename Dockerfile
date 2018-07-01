FROM arwineap/docker-ubuntu-python3.6

RUN apt-get update

# Install latest version of python3
RUN apt install -y python3.6
RUN pip install --upgrade pip

# Install bindsnet and dependencies
RUN pip install bindsnet

RUN apt-get install -y python3-tk
RUN apt install -y libglib2.0-0
RUN apt install -y libsm6 libxext6

# Install vim
RUN apt install -y vim 
RUN pip install jupyter -U && pip install jupyterlab
