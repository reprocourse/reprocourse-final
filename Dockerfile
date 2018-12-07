FROM ubuntu:18.04

RUN apt-get update -qqq &&\
    apt-get install --no-install-recommends -y python3.7 python3-pip python3-setuptools git &&\
    rm -rf /var/lib/apt/lists/* &&\
    pip3 install --no-cache-dir numpy matplotlib tqdm nilearn networkx &&\
    git clone https://github.com/FIU-Neuro/brainconn.git &&\
    cd brainconn &&\
    python3 setup.py install &&\
    cd .. &&\
    rm -R brainconn
    
ENTRYPOINT ["bash"]
