# Databricks notebook source
# MAGIC %pip install google-api-python-client google-auth google-auth-httplib2 google-auth-oauthlib

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
import os
import shutil

# Define paths for email contents and output folder
email_contents_folder = "/Volumes/workspace/default/shortlist_ai/email_contents/"
output_folder = "/Volumes/workspace/default/shortlist_ai/output/"

# Create email contents folder
if os.path.exists(email_contents_folder):
    shutil.rmtree(email_contents_folder)
os.makedirs(email_contents_folder) 

# Function to read the job title from the job description file
def read_job_title(job_description_file):
    with open(job_description_file, 'r') as file:
        job_title = file.readline().strip()
    return job_title

# Function to parse the output file and extract candidate information
def parse_output_file(output_file):
    candidates = []
    with open(output_file, 'r') as file:
        for line in file:
            if line.startswith("Ranked Resumes:"):
                continue

            if line.strip() == "":
                continue

            line_parts = line.split(" - ")
            if len(line_parts) >= 2:
                try:
                    name_part = line_parts[1].split(", Email: ")[0].replace("Name: ", "").strip()
                    email_part = line_parts[1].split(", Email: ")[1].split(", Score: ")[0].strip()
                    score_part = line_parts[1].split(", Score: ")[1].split(", Reason: ")[0].strip()
                    reason_part = line_parts[1].split(", Score: ")[1].replace("Reason: ", "").strip()

                    candidates.append({
                        "name": name_part,
                        "email": email_part,
                        "score": float(score_part.split('/')[0]),
                        "reason": reason_part
                    })
                except (IndexError, ValueError) as e:
                    print(f"Error parsing line: {line.strip()} - {e}")
    return candidates

# Function to generate questions for the job using the LLM
def generate_questions_for_job_using_llm(job_title):
    w = WorkspaceClient()
    prompt = f"""
    You are an HR assistant tasked with creating a questionnaire for candidates applying for the {job_title} role. 
    Based on the responsibilities and skills necessary for the job title, generate a list of thoughtful, open-ended questions that will help assess the candidate's qualifications and fit for the position.
    The questions must no longer than one line. Minimum of 5 and maximum of 10 questions.
    """

    response = w.serving_endpoints.query(
        name="databricks-mixtral-8x7b-instruct",
        messages=[ChatMessage(role=ChatMessageRole.USER, content=prompt)]
    )
    
    questions = response.choices[0].message.content
    return questions

# Function to generate the email content for a candidate
def generate_email_using_llm(name, reason, job_title, score):
    w = WorkspaceClient()
    prompt = f"""
    You are a helpful HR assistant. Based on the following details, write an email to a candidate.

    Candidate Name: {name}
    Job Title: {job_title}
   
    Reason for selection: {reason}

    Make the email friendly, professional, and personalized. Also, mention that the candidate should expect an interview invitation soon.
    """

    response = w.serving_endpoints.query(
        name="databricks-mixtral-8x7b-instruct",
        messages=[ChatMessage(role=ChatMessageRole.USER, content=prompt)]
    )
    
    email_content = response.choices[0].message.content
    return email_content

if __name__ == "__main__":
    job_description_file = r'/Volumes/workspace/default/shortlist_ai/output/job_description.txt'
    output_file = r'/Volumes/workspace/default/shortlist_ai/output/ranked_resumes.txt'

    job_title = read_job_title(job_description_file)
    print(f"Job Title: {job_title}")

    candidates = parse_output_file(output_file)
    print(f"Parsed Candidates: {candidates}")

    score_threshold = 7.0 

    # Generate form questions based on the job description
    form_questions = generate_questions_for_job_using_llm(job_title)

    # Send emails to shortlisted candidates
    for candidate in candidates:
        name = candidate['name']
        reason = candidate['reason']
        email = candidate['email']
        score = candidate['score']

        file_path = os.path.join(email_contents_folder, f"{name}_email_content.txt")
        if score >= score_threshold:
            email_content = generate_email_using_llm(name, reason, job_title, score)
            with open(file_path, "w") as file:
                file.write(f"Email to {name} ({email}):\n")
                file.write(email_content + "\n")
                file.write("-" * 80 + "\n")
        
    print("Email content saved successfully")

