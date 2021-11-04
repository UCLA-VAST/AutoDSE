"""
Design Space Generator
"""
from logging import Logger
from os.path import basename, dirname, join, exists
from subprocess import Popen, PIPE
from glob import iglob
from shutil import copy
import json
from os import remove

from ..logger import get_default_logger


def get_dsgen_logger() -> Logger:
    """ Attach the logger of this module """
    return get_default_logger('DSGen')

def ds_gen(src_file) -> str:
    """ Generate the candidate desgin space 

    Args:
        src_file: tha path to the kernel source file
    Generates:
        src_file: Modified version of the kernel file with the candidate pragmas
        ds_info.json: design space information
    """
    log = get_dsgen_logger()
    
    kernel_file, src_dir = basename(src_file), dirname(src_file)
    
    ## Run ds_generator
    p = Popen(f"cd {src_dir} \n ds_generator {kernel_file}", shell = True, stdout = PIPE)
    p.wait()
    
    
    if exists(join(src_dir, 'ds_info.json')):
        kernel_with_ds = [f for f in iglob(join(src_dir, '*'), recursive=True) if f.startswith('rose_merlinkernel_')][0]
        copy(kernel_with_ds, src_file)
        ds_file = join(src_dir, 'ds_info.json')
        with open(ds_file, 'r') as f_ds:
            ds_info = json.load(f_ds)
            ds_info['timeout.exploration'] = 1200 
            ds_info['timeout.hls'] = 80 
            ds_info['timeout.transform'] = 20 
        remove(ds_file)
        with open(ds_file, 'w') as f_ds:
            json.dump(ds_info, f_ds, indent=4)
        
        log.info(f'Added candidate pragmas.')
        
        return ds_file
    else:
        log.error(f'Failed to generate the database. Check ds_generator log files.')
        
        return None
    
    
