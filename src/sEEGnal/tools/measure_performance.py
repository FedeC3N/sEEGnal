#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Measure the use of memory and the time of execution

Created on Thu 20/11/2024

@author: Fede
"""

# Imports
import os
import json
import time
import tracemalloc
from datetime import datetime as dt, timezone

import sEEGnal.io.bids as bids


def export_execution_results(config,bids_path,elapsed_time,mem_usage,process_name):

    # Create the derivative path
    json_file = bids.build_derivative(bids_path,'desc-measure_performance.json')

    if os.path.exists(json_file):

        # Read the info
        with json_file.open('r') as file:
            json_data = json.load(file)

        # Update the info
        now = dt.now(timezone.utc)
        formatted_now = now.strftime("%d-%m-%Y %H:%M:%S")
        json_data[process_name] = {
            'times_seconds' : elapsed_time,
            'memory_bytes' : mem_usage,
            'date': formatted_now
        }

    else:

        # Creates the JSON dictionary.
        now = dt.now(timezone.utc)
        formatted_now = now.strftime("%d-%m-%Y %H:%M:%S")
        json_data = {
            process_name:{
                'times_seconds': elapsed_time,
                'memory_bytes': mem_usage,
                'date': formatted_now
            }
        }

    # Write the json file
    destination_folder = os.path.abspath(os.path.dirname(json_file))
    if os.path.exists(destination_folder) == False:
        os.makedirs(destination_folder)
    with open(json_file,'w') as fp:
        json.dump(json_data,fp,indent=4)



def measure_performance(func):

    def wrapper(*args):

        # Start measuring memory and time
        start = time.time()
        tracemalloc.start()

        if len(args) == 2:
            config = args[0]
            bids_path = args[1]
            results = func(config,bids_path)

        elif len(args) == 3:
            config = args[0]
            current_file = args[1]
            bids_path = args[2]
            results = func(config,current_file,bids_path)

        # Estimate performance
        end = time.time()
        base,peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        elapsed_time = end - start
        mem_usage = peak - base

        # Export the ETL performance results
        export_execution_results(config,bids_path,elapsed_time,mem_usage,func.__name__)

        return results

    return wrapper



