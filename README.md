# reprocourse-final

## Description
This is the final project for NEUR608 Repoducibility in Neuroscience (18 Fall)
- The final paper and tex files are in `Documents` folder, along with the in-class presentation file.
- The python code for reproducing the figures are in `Command-Files` folder
- Data from original source and produced as cache in the analysis are stored in `Original-Data` and `Analysis-Data` each.

## Requirements
- Python 3.6 or above
- numpy, matplotlib, tqdm, brainconn, networkx, nilearn

## How to reproduce
- Install the required python version and packages, I included a Dockerfile only to restore the environment (as docker is not suitable for generating figures interactively).
- Run `main.py` in `Command-Files`, in the main script, comment / uncomment each line to get figures from the final paper
- For detailed description of each function, see the source code for comments.