"""resources.py - resources for testing are placed here."""

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
    'show_ci': bool,
    'confidence_value': float,
    'must_remove_outliers': bool,
    'show_violin': bool
}
