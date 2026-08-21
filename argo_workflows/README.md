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

# Running Using the nrt-pipeline-templates

To avoid needing to create and monitor many separate cron jobs, we have created the following files under 
agro workflows

/workflow_templates/nrt-pipeline-templates.yaml
/near_real_time/cron_jobs/pipeline/nrt-pipeline-cron.yaml

The file nrt-pipeline-templates.yaml contains and defines all steps needed, as well as providing a DAG for dependencies and logic for retrying missing steps.
To run this, first apply

kubectl -n argo apply -f nrt-pipeline-templates.yaml

and then do

kubectl -n argo apply -f nrt-pipeline-cron.yaml 

the cron jobs will be created paused, you will need to resume, or submit


# Notes on the Regions

Canada is a large region with many lakes. For this reason, like Eurasia, we are splitting Canada into 
Canada1, 2, 3, and 4. By running both downloading and processing for Canada in parallel with smaller regions, it takes much less time.

## Notes on the order

The steps for running the pipeline are as follows

1. download dynamic world data (in parallel)
2. merge each region's new dynamic world data into a single date stamped file.
3. process using the NRT Breakpoint for each region
4. Combine the results into a single zarr archive
5. Combine the results and all historical ones together into 1 zarr archive.
6. Combine the historical dynamic world data and the newly downloaded data into a new netcdf file
7. Upload both the input (the dynamic world data) and the output, both by region, and by file.

## Where to Find New Output

The new output (for now) will be located in this google cloud bucket

Output by Region and Dates Here
gs://pdg-storage-default/water-timeseries-v2/data/output

Dynamic world data here:
gs://pdg-storage-default/water-timeseries-v2/near-real-time/dynamic_world_data

Zarr archive (complete NRT results) here:
gs://pdg-storage-default/water-timeseries-v2/near-real-time/output

# YAML Files That Need To Be Applied

Below, this explains the previous method of using individual yaml files. These may be removed in the future
as this is deprecated, but for now, they have been left in. 

## Sync Bucket with the Latest Output

kubectl -n argo apply -f upload/upload.yaml

Make sure this is running first, it will check the output directory at regular intervals
and sync the bucket with the new contents

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

These will continue to run and retry if there are downloading failures for
any region. Additionally, we found that some zones would get stuck showing 99% completion, so their rule is that,
if after a retry, the zone is still 99% complete, it considers it finished.

## Merging the partial results for a region into a single netcdf file

kubectl -n argo apply -f merge/merge_new_region_results.yaml

This merges if the downloading is finished. 

## Running the NRT Pipeline

kubetl -n argo apply -f process/process_alaska.yaml
kubetl -n argo apply -f process/process_test.yaml
kubetl -n argo apply -f process/process_canada1.yaml
kubetl -n argo apply -f process/process_canada2.yaml
kubetl -n argo apply -f process/process_canada3.yaml
kubetl -n argo apply -f process/process_canada4.yaml
kubectl -n argo apply -f process/process_eurasia1.yaml
kubectl -n argo apply -f process/process_eurasia2.yaml
kubectl -n argo apply -f process/process_eurasia3.yaml

These jobs check for the new merged netcdf file that contains all new data
for a region and the latest summer month. If that file exists and is verified
to contain all downloaded data, it will process, if the processing is not finished.
This also can restart and retry on a fail, saving intermediate progress.

The output of this is the zarr dataset for the region/month. 

## Create New Historical File

kubectl -n argo apply -f merge/create_new_historical_file.yaml

This latest historical dynamic world file will combine existing historical data 
and all newly downloaded data, once the downloading and merging is done. This is meant to run after
processing, so that scripts do not end up using different input files. 

