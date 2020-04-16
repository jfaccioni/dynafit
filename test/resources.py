SETTINGS_SCHEMA = {
    'filename': str,
    'sheetname': str,
    'must_calculate_growth_rate': bool,
    'time_delta': float,
    'cs_start_cell': str,
    'cs_end_cell': str,
    'gr_start_cell': str,
    'gr_end_cell': str,
    'individual_colonies': int,
    'large_colony_groups': int,
    'bootstrap_repeats': int,
    'add_confidence_interval': bool,
    'confidence_value': float,
    'must_remove_outliers': bool,
    'add_violin': bool
}
