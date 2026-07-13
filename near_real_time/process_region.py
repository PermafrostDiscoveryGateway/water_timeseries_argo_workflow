def process_summer_months_for_region(
        region: str,
        years: List[int],
        env_path: str = None,
        force_processing: bool = False,
        use_direct_method: bool = True
) -> Dict[str, Any]:
    """
    Process summer months (June-September) for a region over multiple years.
    Processes dates in reverse chronological order (latest first).
    Uses process_region_date which handles both historical and downloaded data.
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"PROCESSING SUMMER MONTHS FOR REGION: {region}")
    logger.info(f"{'=' * 80}")
    logger.info(f"Years to process: {years}")
    logger.info(f"Using direct processing method: {use_direct_method}")
    logger.info(f"Processing order: Latest to oldest (reverse chronological)")

    if env_path:
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()

    results = {
        'region': region,
        'years': years,
        'months_processed': [],
        'results': {},
        'use_direct_method': use_direct_method
    }

    # Build list of all months to process
    all_months = []
    for year in years:
        summer_months = get_summer_months(year)
        summer_dates = get_summer_dates_for_processing(year)
        for month_str, timestamp in zip(summer_months, summer_dates):
            all_months.append((year, month_str, timestamp))

    # Sort by date in reverse order (latest first)
    all_months.sort(key=lambda x: x[2], reverse=True)

    logger.info(f"Total months to process: {len(all_months)}")
    logger.info(f"Processing order: {[m[1] for m in all_months]}")

    for year, month_str, timestamp in all_months:
        logger.info(f"\nChecking month: {month_str}")

        # Check if data is available (checks BOTH historical and merge)
        availability = check_data_availability_for_date(region, month_str, env_path)

        if not availability.get('available', False):
            logger.warning(f"  ⚠️ No data available for {region} {month_str}")
            logger.warning(f"     Reason: {availability.get('message', availability.get('error', 'Unknown'))}")
            logger.info(f"     Note: Historical data for {month_str} may exist in the compressed file")

            results['results'][month_str] = {
                'success': False,
                'reason': 'No data available',
                'details': availability,
                'year': year,
                'month': month_str,
                'timestamp': timestamp
            }
            continue

        # Log the source of the data
        source = availability.get('source', 'unknown')
        logger.info(f"  ✅ Data available for {region} {month_str} (source: {source})")
        logger.info(f"     IDs in file: {availability.get('id_count', 0):,}")

        # Process the month
        try:
            logger.info(f"  Processing {region} for {month_str}...")

            process_result = process_region_date_new(
                region=region,
                analysis_date=month_str,
                env_path=env_path,
                batch_size=1000
            )

            results['results'][month_str] = {
                'success': process_result.get('success', False),
                'timestamp': timestamp,
                'year': year,
                'month': month_str,
                'availability': availability,
                'analysis_source': process_result.get('analysis_source', 'unknown'),
                'total_ids': process_result.get('total_ids', 0),
                'processed': process_result.get('processed', 0),
                'breakpoints_found': process_result.get('breakpoints_found', 0),
                'zarr_path': process_result.get('zarr_path', None)
            }

            if results['results'][month_str]['success']:
                logger.info(f"  ✅ Successfully processed {region} {month_str}")
                results['months_processed'].append(month_str)
            else:
                logger.warning(f"  ❌ Failed to process {region} {month_str}")

        except Exception as e:
            logger.error(f"  ❌ Error processing {region} {month_str}: {e}")
            import traceback
            traceback.print_exc()
            results['results'][month_str] = {
                'success': False,
                'error': str(e),
                'timestamp': timestamp,
                'year': year,
                'month': month_str,
                'availability': availability
            }

        # Small delay between processing months to avoid memory issues
        time.sleep(2)

    # Summary for the region
    processed_count = sum(1 for r in results['results'].values() if r.get('success', False))
    total_count = len(results['results'])

    logger.info(f"\n{'=' * 80}")
    logger.info(f"SUMMARY FOR REGION: {region}")
    logger.info(f"{'=' * 80}")
    logger.info(f"Total months processed: {total_count}")
    logger.info(f"Successfully processed: {processed_count}")
    logger.info(f"Failed/Skipped: {total_count - processed_count}")

    if processed_count > 0:
        success_rate = (processed_count / total_count) * 100
        logger.info(f"Success rate: {success_rate:.1f}%")

        # Show breakpoints found
        total_breakpoints = sum(
            r.get('breakpoints_found', 0)
            for r in results['results'].values()
            if r.get('success', False)
        )
        if total_breakpoints > 0:
            logger.info(f"Total breakpoints found: {total_breakpoints:,}")

    return results