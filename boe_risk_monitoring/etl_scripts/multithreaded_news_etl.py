from concurrent.futures import ThreadPoolExecutor, as_completed
import os

from boe_risk_monitoring.etl_scripts.etl_classes import FinancialNewsETL
from boe_risk_monitoring.utils.utils import construct_input_output_file_paths_news
import boe_risk_monitoring.config as config

NEWS_DATA_FOLDER_PATH = config.NEWS_DATA_FOLDER_PATH

def process_news_article(input_pdf_path, output_dir_path, llm_backend="gemini", llm_model_name="gemini-2.5-pro-preview-06-05"):

	if not isinstance(input_pdf_path, str):
		raise TypeError("Input path must be of type str.")

	if not isinstance(output_dir_path, str):
		raise TypeError("Output directory path must be of type str.")

	financial_news_etl = FinancialNewsETL(
		input_pdf_path=input_pdf_path,
	)

	# Run the transform method (no need to run extract method for presentations)
	analysis_results_dict = financial_news_etl.transform(
		llm_backend=llm_backend,
		llm_model_name=llm_model_name,
	)

	# Run the load method
	financial_news_etl.load(
		transformed_data=analysis_results_dict,
		output_dir_path=output_dir_path,
	)

if __name__ == "__main__":
	# Get the input and output file paths
	input_pdf_paths, output_dirs = construct_input_output_file_paths_news(news_dir_path=NEWS_DATA_FOLDER_PATH, skip_if_output_exists=True)
	print(f"Found {len(input_pdf_paths)} news article files to process.")

	# input_pdf_paths = input_pdf_paths[:2]  # Limit to first 10 files for testing
	# output_dirs = output_dirs[:2]  # Limit to first 10 directories for testing

	# Create a ThreadPoolExecutor to process files concurrently
	with ThreadPoolExecutor(max_workers=4) as executor:
		futures = {}
		for input_pdf_path, output_dir_path in zip(input_pdf_paths, output_dirs):
			future = executor.submit(
				process_news_article,
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



