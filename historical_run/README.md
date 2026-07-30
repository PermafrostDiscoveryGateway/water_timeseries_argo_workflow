## Historical Run Instructions

Make sure that you have a .env file with the values that are in example.env

These values

`region_name=TEST
start_year=2023
start_month=6
end_year=2023
end_month=7`

You are the ones that determine dates and region.

There are optional .env variables related to processing and Rbeast parameters, but they are 
set to defaults (same as within the script)

To run, simply run

`python process_historical_BST.py`