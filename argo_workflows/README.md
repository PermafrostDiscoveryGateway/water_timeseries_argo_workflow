## Explanation of the Argo Workflows

This document explains the NRT pipeline 

# Running the Near Real Time

Running the near real time pipeline involves the following steps

1. During summer months, download data for the last complete month for each region.
2. Merge the partial download files for each region into a merged netcdf file.
3. Once the merged file for each region and date is finished, process using the NRT Breakpoint.
4. Periodically sync the google cloud bucket with the results in the output directory.
5. Create a new historical file that includes the new data.

# Notes on The Order

The downloading of data is a time consuming step. For this reason, we have regions download in parallel, and then 
a merged netcdf file for the region and date is created as soon as that region is done downloading. Once
this file is created, we can process and run the NRT breakpoint for that region and the latest month. This way we can run the NRT breakpoint 
on a region as soon as that region is downloaded and the new data is merged into a netcdf file, even if other regions
are still downloading. 

# Notes on the Regions

Canada is a large region with many lakes. For this reason, like Eurasia, we are splitting Canada into 
Canada1, 2, 3, and 4. By running both downloading and processing for Canada in parallel with smaller regions, it takes much less time.

# YAML Files That Need To Be Applied

## Sync Bucket with the Latest Output

## Downloading

Apply the following files

kubetl -n argo apply -f download/download_alaska.yaml
kubetl -n argo apply -f download/download_test.yaml
kubetl -n argo apply -f download/download_canada1.yaml
kubetl -n argo apply -f download/download_canada2.yaml
kubetl -n argo apply -f download/download_canada3.yaml
kubetl -n argo apply -f download/download_canada4.yaml
kubectl -n argo apply -f download/download_eurasia1.yaml
kubectl -n argo apply -f download/download_eurasia2.yaml
kubectl -n argo apply -f download/download_eurasia3.yaml

## Merging the partial results for a region into a single netcdf file

## Running the NRT Pipeline

## Create New Historical File