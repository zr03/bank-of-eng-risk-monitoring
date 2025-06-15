from concurrent.futures import ThreadPoolExecutor, as_completed
import os

from boe_risk_monitoring.etl_scripts.etl_classes import PresentationETL
from boe_risk_monitoring.utils.utils import construct_input_output_file_paths
import boe_risk_monitoring.config as config

DATA_FOLDER = config.DATA_FOLDER

def process_presentation(input_pdf_path, output_dir_path, llm_backend="gemini", llm_model_name="gemini-2.5-pro-preview-06-05"):

	if not isinstance(input_pdf_path, str):
		raise TypeError("Input path must be of type str.")

	if not isinstance(output_dir_path, str):
		raise TypeError("Output directory path must be of type str.")

	# Identify if this is a Q4 presentation
	if "Q4" in os.path.basename(input_pdf_path):
		print(f"Identified {input_pdf_path} as a Q4 presentation.")
		is_q4_presentation = True
	else:
		print(f"Identified {input_pdf_path} as a non-Q4 presentation.")
		is_q4_presentation = False

	presentation_etl = PresentationETL(
		input_pdf_path=input_pdf_path,
		is_q4_presentation=is_q4_presentation,
	)

	# Run the transform method (no need to run extract method for presentations)
	analysis_results_dict = presentation_etl.transform(
		llm_backend=llm_backend,
		llm_model_name=llm_model_name,
	)

	# Run the load method
	presentation_etl.load(
		transformed_data=analysis_results_dict,
		output_dir_path=output_dir_path,
	)

if __name__ == "__main__":
	# Get the input and output file paths
	input_pdf_paths, output_dirs = construct_input_output_file_paths(data_dir=DATA_FOLDER, input_file_type="presentations", skip_if_output_exists=True)
	print(f"Found {len(input_pdf_paths)} presentation files to process.")


	# Create a ThreadPoolExecutor to process files concurrently
	with ThreadPoolExecutor(max_workers=4) as executor:
		futures = {}
		for input_pdf_path, output_dir_path in zip(input_pdf_paths, output_dirs):
			future = executor.submit(
				process_presentation,
				input_pdf_path=input_pdf_path,
				output_dir_path=output_dir_path,
				llm_backend="gemini",
				llm_model_name="gemini-2.5-pro-preview-06-05"
			)
			futures[future] = input_pdf_path
		print(f"Submitted {len(futures)} tasks for processing.")

		for future in as_completed(futures):
			input_pdf_path = futures[future]
			try:
				future.result()  # This will raise an exception if the processing failed
			except Exception as e:
				print(f"Error processing file {input_pdf_path}: {e}")
			else:
				print(f"Processed presentation {input_pdf_path} successfully.")



