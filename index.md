# Design Space Definition

The candidate pragmas for AutoDSE can be in either of the following forms:

````C
#pragma ACCEL PIPELINE auto{pragma_name}
#pragma ACCEL PARALLEL factor=auto{pragma_name}
#pragma ACCEL TILE factor=auto{pragma_name}
````

You can either let AutoDSE augment the code with the candidate pragmas or do it yourself. To make AutoDSE do it, run one of the following commands based on your use case:

````bash
ds_generator [-I<include dir>] <kernel file> # only adds the candidate pragmas to the kernel code and stores the generated code in `rose_merlinkernel_[KERNEL_NAME].c`

autodse <project dir> <working dir> <kernel file> <fastgen|accurategen> [<database file>]  # adds candidate pragmas and runs DSE after it
````

Each of the above commands will produce a file named `ds_info.json` which describes the design space along with AutoDSE's settings. Each of the pragmas would be defined in the following form:

````json
"__PARA__L0" : 
		{
			"ds_type" : "PARALLEL",
			"options" : "[x for x in [1,2,4,7,8,14,16,32] if x*__TILE__L0<=32]",
            "default" : 1,
			"order" : "0 if x&(x-1)==0 else 1"
		},
````

- **ds_type:** this attribute defines the type of the pragma. Choices are `PIPELINE`, `PARALLEL`, and `TILE`.
- **options:** this attribute shows the available options for the respective pragma as a Python list. As the example shows, it can have a condition part to define a dependency with other pragmas. Naturally, the definition shouldn't result in an empty test.
- **default:** the default option for the pragma which should turn it off and be an element of the `options` attribute.
- **order:** this attribute can describe a preference in searching order for the respective pragma. In the above example, we set higher priority to the values that are powers of 2.

Please refer to [this file](https://github.com/UCLA-VAST/AutoDSE/blob/gh-pages/test/ds_info.json) for a complete example on design space (DS) definition for the [GEMM kernel from the Machsuite benchmark](https://github.com/breagen/MachSuite/tree/master/gemm/ncubed)

# AutoDSE Settings
The rest of the settings in the [config file](https://github.com/UCLA-VAST/AutoDSE/blob/gh-pages/test/ds_info.json) determines how AutoDSE should be run. The following table gives a description of each of them.

| Setting | Description|
| :---        |    :----   |
| design-space.max-part-num | the maximum numbers of DS partitions allowed |
| evaluate.command.bitgen | Merlin command to generate the bitstream |
| evaluate.command.hls | Merlin command to run the HLS synthesis |
| evaluate.command.transform | Merlin command for applying its code transformations |
| evaluate.max-util.[XX] | the maximum allowed utilization for resource XX |
| project.backup | type of project backup, choices: `BACKUP_ERROR`, `NO_BACKUP`, or `BACKUP_ALL`
| project.fast-output-num | number of top designs generated as Merlin projects in fast mode |
| search.algorithm.gradient.fine-grained-first | if set to `true` starts optimizing from the innermost loops in the bottleneck optimizer |
| search.algorithm.gradient.latency-threshold | the minimum threshold latency we wish to achieve |
| search.algorithm.gradient.quality-type | how to measure the quality of design, choices: `performance`, or `finite-difference`
| search.algorithm.name | exploration strategy, use `gradient` to implement the paper's approach. You can also use `exhaustive` for exhaustive search or `hybrid` for a combination of these two.
| timeout.bitgen | time limit (in minutes) for generating the bitstream |
| timeout.exploration | time limit (in minutes) for DSE |
| timeout.hls | time limit (in minutes) for the HLS synthesis |
| timeout.transform |  time limit (in minutes) for Merlin's code transformation |

# Output Structure
The generated files structure is as shown below:
````
.
+-- evaluate          # backup directory which includes some of the explored Merlin projects

+-- output            # includes the Merlin project for top designs and a summary of them 

+-- logs              # all the log files
|   +-- dse.log         # contains all the messages printed in the console
|   +-- eval.log        # shows how many jobs were run and whether or not they were successful
|   +-- partX_log.log   # includes a summary of the status of explorer in DS partition X
|   +-- partX_expr.log  # includes a more detailed summary of the explorer in DS partition X

+-- result.db        # includes a database of the explored designs

+-- summary_[XX].rpt # summarizes all the points explored in mode XX (`fast` or `accurate`)
````

# General Tips
1. When AutoDSE is finished, it will save all the explored design points in a file (by default: `redis.db`) as a [Redis database](https://developer.redis.com/develop/python/). In the presence of such file, you can resume the DSE by re-running the DSE command:

````bash
dse <project dir> <working dir> <config file> <fast|accurate> [<database file>]
````

2. Since AutoDSE runs the HLS tool to assess a design point, the timeout values may need to be changed across different kernels. If the final report of the AutoDSE shows high number of timed out designs, adjust the respective time limit and rerun the tool. 