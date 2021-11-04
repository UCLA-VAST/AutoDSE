# AutoDSE

## Publication

+ Atefeh Sohrabizadeh, Cody Hao Yu, Min Gao, Jason Cong. [AutoDSE: Enabling Software Programmers to Design Efficient FPGA Accelerators](https://arxiv.org/abs/2009.14381). In ACM TODAES, 2021.

## About
This repo contains the codes for AutoDSE which can help you optimize your FPGA design. For more information on what AutoDSE is please refer to the [publication](https://arxiv.org/abs/2009.14381).


## Content
1. [Requirements and Dependencies](#requirements-and-dependencies)
2. [Project File Tree](#project-file-tree)
3. [Run AutoDSE](#run-autodse)
4. [Citation](#citation)


## Requirements and Dependencies
### Development Requirements
This project is built on top of the [**Merlin Compiler**](https://github.com/Xilinx/merlin-compiler). You should install that first.

For testing and deployment, you also need to have at least one Xilinx tool (SDaccel or Vitis) installed.

Lastly, you need to install [docker](https://docs.docker.com/get-docker/).


## Project File Tree
The project file structure is shown below,
````
.
+-- autodse # AutoDSE source codes in Python
+-- docker  # Files needed for installing AutoDSE in a docker
````

## Run AutoDSE
### Installing the Project
If you have installed the [dependencies](#requirements-and-dependencies), you can build the docker containing the AutoDSE project using:
````bash
cd docker
./docker-build.sh
````

### Invoking AutoDSE
1. Invoke the installed docker in an interactive session:
````bash
cd docker
./docker-run.sh -i /bin/bash
````
2. You can run AutoDSE in any of the following forms depending on your use case. 

**Note:** Remember that before proceeding with this part, you should make sure that your source directory runs with the Merlin Compiler. 

#### Design Space Generator + Explorer
If you want to run AutoDSE through all the steps of augmenting the kernel code with candidate pragmas and running an explorer on it, run the following command:
````bash
autodse <project dir> <working dir> <kernel file> <fastgen|accurategen> [<database file>]
````

#### Desgin Space Generator
If you only want to augment the code with the candidate pragmas and analyze them, run the following command:
````bash
ds_generator [-I<include dir>] <kernel file>
````

#### Design Space Explorer
If you already have defined your design space and augmented the code with the candidate pragmas (either using AutoDSE or writing your own files) and only want to run the explorer, run the following command:
````bash
dse <project dir> <working dir> <config file> <fast|accurate> [<database file>]
````


## Citation
If you find any of the ideas/codes useful for your research, please cite our paper:

	@article{sohrabizadeh2020autodse,
  		title={AutoDSE: Enabling Software Programmers Design Efficient FPGA Accelerators},
  		author={Sohrabizadeh, Atefeh and Yu, Cody Hao and Gao, Min and Cong, Jason},
 		journal={arXiv preprint arXiv:2009.14381},
  		year={2020}
	}


