from pathlib import Path

import boe_risk_monitoring.config as config

DATA_FOLDER = config.DATA_FOLDER
PERMISSIBLE_INPUT_FILE_TYPES = ["transcripts", "presentations"]
AGGREGATED_DATA_FOLDER_NAME = config.AGGREGATED_DATA_FOLDER_NAME

def get_bank_dirs(data_dir=DATA_FOLDER):
	data_dir = Path(data_dir)
	bank_dirs = [bank_dir for bank_dir in data_dir.iterdir() if bank_dir.is_dir() and bank_dir.name != AGGREGATED_DATA_FOLDER_NAME]
	return bank_dirs


def construct_input_output_file_paths(data_dir=DATA_FOLDER, input_file_type="transcripts", skip_if_output_exists=False):
	if input_file_type not in PERMISSIBLE_INPUT_FILE_TYPES:
		raise ValueError(f"Input file type must be one of {PERMISSIBLE_INPUT_FILE_TYPES}. Got {input_file_type} instead.")
	bank_dirs = get_bank_dirs(data_dir)
	input_dirs = []
	output_dirs = []
	for bank_dir in bank_dirs:
		input_dir = Path(bank_dir, "raw_docs", input_file_type)
		if not input_dir.exists():
			print(f"Skipping bank {bank_dir.name} as input directory does not exist.")
			continue
		output_dir = Path(bank_dir, "processed", input_file_type)
		output_dir.mkdir(parents=True, exist_ok=True)
		input_dirs.append(input_dir)
		output_dirs.append(output_dir)

	input_fpaths = []
	output_dirs_expanded = []
	for i, input_dir in enumerate(input_dirs):
		if not input_dir.exists():
			print(f"Input directory {input_dir} does not exist. Skipping.")
			continue
		# Get pdf files in the input directory
		presentation_fpaths = list(input_dir.glob("*.pdf"))
		if not presentation_fpaths:
			print(f"No presentation files found in {input_dir}. Skipping.")
			continue
		output_objs = [output_obj for output_obj in output_dirs[i].iterdir() if output_obj.is_file()]
		for presentation_fpath in presentation_fpaths:
			if skip_if_output_exists:
				# Check if the output file already exists
				fname = presentation_fpath.stem
				fname_comps = fname.split("_")
				reporting_period = fname_comps[0] + "_" + fname_comps[1] # Assuming the first two components are the reporting period
				# Get list of files in the output directory that match the reporting period
				output_files_reporting_period = [output_obj for output_obj in output_objs if output_obj.stem.startswith(reporting_period)]
				if output_files_reporting_period:
					print(f"Skipping {presentation_fpath} as output already exists in {output_dirs[i]}.")
					continue

			input_fpaths.append(str(presentation_fpath)) # Convert to string for downstream compatibility
			output_dirs_expanded.append(str(output_dirs[i])) # Convert to string for downstream compatibility

	if len(input_fpaths) != len(output_dirs_expanded):
		raise ValueError("Something went wrong! Mismatch between number of input files and output directories. Check your input directories.")

	return input_fpaths, output_dirs_expanded
