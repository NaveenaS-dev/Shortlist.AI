# Databricks notebook source
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
import os

def read_input_file(input_file):
    with open(input_file, 'r') as file:
        content = file.read().strip()
    return content

def generate_job_description(file_content):
    # Initialize Databricks Workspace Client
    w = WorkspaceClient()

    # Create the prompt for generating job description
    prompt = (
        f"Based on the following content, generate a detailed job description with two sections:\n{file_content}\n"
        "1. **Our Ideal Fit**: Include required experience, technical skills, and other competencies (based on the job title) in bullet points.\n"
        "2. **Your Typical Work Would Include**: List responsibilities and daily tasks clearly in bullet points.\n"
        "Ensure both sections are clear, concise, and formatted as bullet points without long paragraphs."
    )

    # Query the serving endpoint with a ChatMessage object instead of prompt
    response = w.serving_endpoints.query(
        name="databricks-mixtral-8x7b-instruct",
        messages=[ChatMessage(role=ChatMessageRole.USER, content=prompt)]
    )

    # Extract the text from the model's response
    job_description = response.choices[0].message.content
    return job_description

def write_output_to_file(output_file, job_description):
    output_dir = os.path.dirname(output_file)
    
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file, 'w') as file:
        file.write(job_description)

def main(input_file, output_file):
    # Read content from input file
    file_content = read_input_file(input_file)
    
    # Generate job description using the Databricks model
    job_description = generate_job_description(file_content)
    
    # Write the job description to the output file
    write_output_to_file(output_file, job_description)
    
    print("\nJob description generated and saved to:", output_file)

if __name__ == "__main__":
    input_file = "/Volumes/workspace/default/shortlist_ai/input.txt"  # Adjust the path for Databricks DBFS
    output_file = "/Volumes/workspace/default/shortlist_ai/output/job_description.txt"
    main(input_file, output_file)

