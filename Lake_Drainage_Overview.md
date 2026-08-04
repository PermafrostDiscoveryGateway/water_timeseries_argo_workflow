## Overview of the Lake Drainage Pipeline

This repository is for running the lake drainage pipline on the google cloud 
kubernetes cluster using argo workflows. Various steps are handled by different argo workflow 
cron jobs.

## Inputs and outputs

For input, we start with dynamic world data that is both from a historical netcdf file, and that
is downloaded in new netcdf files for each grid tile, region, and month. The pipeline merges this new data into a new netcdf file.

Output will be in a .zarr dataset. Each region/date run will be in its own zarr dataset.

## Steps in the pipeline

To allow for more parallel processing, the argo workflow is divided into multiple parts. 

1. Downloading monthly date from dynamic world (regions run in parallel). Before downloading, this checks to see if the download previously ran, and if it did, did it finish. Download is marked complete when we have over 99% of expected grid tiles downloaded. This was chosen since, occasionally, a few grid tiles fail to download.
2. Merging newly downloaded data into a new netcdf file (one job runs). This checks the most recent netcdf file, and then checks the ids and dates of the newly downloaded data to see if it has already been merged. If not, it merges that region. 
3. Running the NRT Breakpoint for the latest date (regions run in parallel). The names of the merged files follow a pattern. This job looks for that file, and that the file is no longer being written to, before running process for each region.
4. Syncing our google cloud bucket with the new output results (cron job that runs periodically). This is in its own cron job, since putting it at the end of process sometimes failed.

Note - every type of job is set up so that, if a job is already running, a new one will not begin. Example - if we are downloading data for Canada
even if the next scheduled time arrives, if the previous job is still completing, the new one will not start. This is because, some steps take a while, but we 
also do not want to wait too long to retry a step if it somehow failed. 

## Following jobs

Jobs can be followed using the argo web interface. You will see the cron jobs under the 'crons'

(insert screenshot here)

If you want to manually submit a cron job you click here

## Running on Autopilot

Autopilot handles scaling of nodes which lowers are costs. However, autopilot does come with various constraints in terms of resources. 

## Where is the data

You can find the data here on this google cloud bucket. 