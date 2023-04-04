# AutoDSE

| [Tutorial](https://ucla-vast.github.io/AutoDSE/) |

## Publication

+ Atefeh Sohrabizadeh, Cody Hao Yu, Min Gao, Jason Cong. [AutoDSE: Enabling Software Programmers to Design Efficient FPGA Accelerators](https://dl.acm.org/doi/full/10.1145/3494534). In ACM TODAES, 2022.

## About
This repo contains the codes for AutoDSE which can help you optimize your FPGA design. For more information on what AutoDSE does please refer to the [publication](https://dl.acm.org/doi/full/10.1145/3494534).

AutoDSE is a fully automated design space exploration that leverages a bottleneck-guided coordinate optimizer to systematically find a better design point. At each iteration, AutoDSE detects the bottleneck of the design and focuses on high-impact parameters to overcome it. 


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

**Note 1:** Remember that before proceeding with this part, you should make sure that your source directory runs with the Merlin Compiler. 

**Note 2:** All the `for` loops should use `{}` for denoting their statements, even if there is only one statement inside the loop.

**Note 3:** The name of the kernel file cannot start with `rose`. 

#### Design Space Generator + Explorer
If you want to run AutoDSE through all the steps of augmenting the kernel code with candidate pragmas and running an explorer on it, run the following command:
````bash
autodse <project dir> <working dir> <kernel file> <fastgen|accurategen> [<database file>]
````
`<project dir>` would be the path to your Merlin project, `<working dir>` is the location in which you want AutoDSE to store the log files, `<kernel file> ` is the path to the C/C++ kernel.
The `fastgen` mode performs DSE based on the HLS synthesis. The `accurategen` mode additionally generates the bitstream and outputs the best HLS design. The `<database file>` is optional. If you don't have a pre-existing database, AutoDSE will create one for you.



#### Desgin Space Generator
If you only want to augment the code with the candidate pragmas and analyze them, run the following command:
````bash
ds_generator [-I<include dir>] <kernel file>
````

The generated file will start with `rose_merlin`. If you want to use this file in the DSE process, replace your kernel file with it and proceed to the [next part](#design-space-explorer). 


#### Design Space Explorer
If you already have defined your design space and augmented the code with the candidate pragmas (either using AutoDSE or writing your own files) and only want to run the explorer, run the following command:
````bash
dse <project dir> <working dir> <config file> <fast|accurate> [<database file>]
````
`<project dir>` would be the path to your Merlin project, `<working dir>` is the location in which you want AutoDSE to store the log files, `<config file>` is the config file (`.json` file) summarizing the design space and AutoDSE's parameters. Refer to [here](https://github.com/UCLA-VAST/AutoDSE/blob/main/examples/gemm-ncubed/sample-dse-files/ds_info.json) to see an example of this file. 
The `fast` mode performs DSE based on the HLS synthesis. The `accurate` mode additionally generates the bitstream and outputs the best HLS design. The `<database file>` is optional. If you don't have a pre-existing database, AutoDSE will create one for you.

## Citation
If you find any of the ideas/codes useful for your research, please cite our paper:

	@article{sohrabizadeh2022autodse,
		title={AutoDSE: Enabling Software Programmers to Design Efficient FPGA Accelerators},
		author={Sohrabizadeh, Atefeh and Yu, Cody Hao and Gao, Min and Cong, Jason},
		journal={ACM Transactions on Design Automation of Electronic Systems (TODAES)},
		volume={27},
		number={4},
		pages={1--27},
		year={2022},
		publisher={ACM New York, NY}
	}


