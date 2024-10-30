# Databricks notebook source
# MAGIC
# MAGIC %pip install --quiet -U transformers==4.41.1 pypdf==4.1.0 langchain-text-splitters==0.2.0 databricks-vectorsearch mlflow tiktoken==0.7.0 
# MAGIC
# MAGIC # %pip install pymupdf 
# MAGIC # %pip install langchain 
# MAGIC # %pip install --upgrade typing_extensions
# MAGIC # dbutils.library.restartPython()
# MAGIC

# COMMAND ----------

# MAGIC %pip install langchain
# MAGIC dbutils.library.restartPython() 

# COMMAND ----------

# MAGIC %md
# MAGIC **Volume setup and PDF storage**
# MAGIC
# MAGIC - Create a volume.
# MAGIC - Upload PDF files to the volume.
# MAGIC - Convert the PDFs to binary format and store them in a Delta table using Autoloader.

# COMMAND ----------

# DBTITLE 1,create  volume in  databricks
# %sql
# CREATE VOLUME IF NOT EXISTS shortlist_ai;

# COMMAND ----------

# DBTITLE 1,add pdf files to volume
'''
import shutil
import os

# Define paths
dbfs_source_folder = "/Workspace/Shared/input files" 
volume_folder = "/Volumes/workspace/default/shortlist_ai"
pdf_folder = volume_folder + "/pdf_files"

# Check if the destination pdf_files folder exists, create it if it does not
if not os.path.exists(pdf_folder):
    os.makedirs(pdf_folder)
    print(f"Created directory: {pdf_folder}")
else:
    print(f"Directory already exists: {pdf_folder}")

# List all PDF files in the DBFS source folder
pdf_files = [f for f in os.listdir(dbfs_source_folder) if f.endswith(".pdf")]

# Copy each PDF file from the DBFS source folder to the volume folder
for pdf_file in pdf_files:
    src_path = os.path.join(dbfs_source_folder, pdf_file)
    dest_path = os.path.join(pdf_folder, pdf_file)
    
    # Copy file
    shutil.copy(src_path, dest_path)
print(f"Copied to {volume_folder}")
'''


# COMMAND ----------

# MAGIC %md
# MAGIC Convert the text to binary format and save it in the table.

# COMMAND ----------

# DBTITLE 1,convert text to binary and store it in table
dbfs_source_folder = "/Workspace/Shared/input files" 
volume_folder = "/Volumes/workspace/default/shortlist_ai"
pdf_folder = volume_folder + "/pdf_files"

df = (spark.readStream
        .format('cloudFiles')
        .option('cloudFiles.format', 'BINARYFILE')
        .option("pathGlobFilter", "*.pdf")
        .load('dbfs:'+volume_folder + "/pdf_files"))

# Write the data as a Delta table
(df.writeStream
  .trigger(availableNow=True)
  .option("checkpointLocation", f'dbfs:{volume_folder}/checkpoints/raw_docs')
  .table('pdf_raw').awaitTermination())

# COMMAND ----------

# MAGIC %md
# MAGIC Bytes to text conversion

# COMMAND ----------

# DBTITLE 1,bytes to text conversion
import warnings
from pypdf import PdfReader  
import io
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Step 1: Load the Delta table from the metastore
df = spark.table("workspace.default.pdf_raw")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      
    chunk_overlap=20    
)

def parse_bytes_pypdf(raw_doc_contents_bytes: bytes):
    try:
        pdf = io.BytesIO(raw_doc_contents_bytes)
        reader = PdfReader(pdf)
        # Extract text from each page and join them into a single string
        parsed_content = [page.extract_text() for page in reader.pages]
        return "\n".join(parsed_content)
    except Exception as e:
        warnings.warn(f"Exception {e} has been thrown during parsing")
        return None

# Step 2: Define a function to convert binary PDF data to cleaned text
def parse_pdf_content(binary_data):
    try:
        # Use parse_bytes_pypdf to extract text from the binary data
        text = parse_bytes_pypdf(binary_data)
        if text:
            # Clean the text
            text = re.sub(r"[\u2022•●▪️►-]", "", text)  # Removes common bullet symbols
            text = re.sub(r"[^a-zA-Z0-9\s.,]", "", text)  # Keeps alphanumeric, space, period, comma
        return text
    except Exception as e:
        warnings.warn(f"Exception {e} has been thrown during parsing")
        return None

# Step 3: Apply the function on each row to extract and clean text from binary content
pdf_chunks = []
for row in df.collect():
    binary_data = row["content"]  
    pdf_text = parse_pdf_content(binary_data) 
    if pdf_text:
        # Create chunks for each PDF's text
        chunks = splitter.split_text(pdf_text)
        pdf_chunks.append(chunks)

# Step 4: Display the cleaned text (or store it in another Delta table if needed)
for i, chunks in enumerate(pdf_chunks):
    print(f"PDF {i + 1} chunks:")
    for j, chunk in enumerate(chunks):
        print(f"Chunk {j + 1}:\n{chunk}\n")


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Create table to store pdf chunks
# MAGIC

# COMMAND ----------

# DBTITLE 1,store chunks in new delta table
# MAGIC %sql
# MAGIC
# MAGIC CREATE TABLE IF NOT EXISTS pdf_text_chunks (
# MAGIC   file_id STRING,              -- Unique identifier for each PDF file
# MAGIC   chunk_index INT,             -- Index of each chunk within the file
# MAGIC   chunk_text STRING,           -- Text content of the chunk
# MAGIC   original_file_path STRING,   -- Path or name to identify the original PDF file
# MAGIC   filename STRING,             -- Name of the original PDF file
# MAGIC   total_chunks INT,            -- Optional: total number of chunks in the file for context
# MAGIC   embeddings ARRAY<FLOAT>,     -- Embedding vector representing the semantic meaning of the chunk
# MAGIC   primary_key STRING GENERATED ALWAYS AS (file_id || '-' || chunk_index)  -- Unique identifier for each chunk
# MAGIC ) USING delta TBLPROPERTIES (delta.enableChangeDataFeed = true);
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC PDF chunks storage in delta table 

# COMMAND ----------

# DBTITLE 1,store chunks after checking token limit
import warnings
from pypdf import PdfReader  
import re
from pyspark.sql import Row
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from transformers import AutoTokenizer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os



# Define the schema explicitly, including filename
schema = StructType([
    StructField("file_id", StringType(), True),
    StructField("chunk_index", IntegerType(), True),
    StructField("chunk_text", StringType(), True),
    StructField("original_file_path", StringType(), True),
    StructField("filename", StringType(), True),  # New field for filename
    StructField("total_chunks", IntegerType(), True)
])

# Load the Delta table from the metastore
df = spark.table("workspace.default.pdf_raw")

# Tokenizer and text splitter setup
tokenizer = AutoTokenizer.from_pretrained("openai-gpt")
max_chunk_size = 500  # Define maximum chunk size by token count
max_token_length = 512  # Maximum token length for the model
splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=tokenizer, 
    chunk_size=max_chunk_size,
    chunk_overlap=50
)

# Function to parse PDF content and clean text
def parse_bytes_pypdf(raw_doc_contents_bytes: bytes):
    try:
        pdf = io.BytesIO(raw_doc_contents_bytes)
        reader = PdfReader(pdf)
        # Extract text from each page and join them into a single string
        parsed_content = [page.extract_text() for page in reader.pages]
        return "\n".join(parsed_content)
    except Exception as e:
        warnings.warn(f"Exception {e} has been thrown during parsing")
        return None

# Step 2: Define a function to convert binary PDF data to cleaned text
def parse_pdf_content(binary_data):
    try:
        # Use parse_bytes_pypdf to extract text from the binary data
        text = parse_bytes_pypdf(binary_data)
        if text:
            # Clean the text
            text = re.sub(r"[\u2022•●▪️►-]", "", text)  # Removes common bullet symbols
            text = re.sub(r"[^a-zA-Z0-9\s.,]", "", text)  # Keeps alphanumeric, space, period, comma
        return text
    except Exception as e:
        warnings.warn(f"Exception {e} has been thrown during parsing")
        return None

# Function to split text based on token size
def tokenize_and_split_text(text, min_chunk_size=20):
    chunks = splitter.split_text(text)
    rows = []  # Prepare to collect rows directly
    
    for chunk in chunks:
        # Immediately split the chunk if it exceeds max_token_length
        while len(tokenizer.encode(chunk)) > max_token_length:
            split_point = max_token_length // 2
            sub_chunk = chunk[:split_point].strip()
            # Check token count for the split chunk
            if len(tokenizer.encode(sub_chunk)) <= max_token_length:
                rows.append(sub_chunk)
                # print(f"Token count for split chunk: {len(tokenizer.encode(sub_chunk))}")
            chunk = chunk[split_point:].strip()  # Reduce chunk to the remaining part

        # Check the token count for the current chunk
        token_count = len(tokenizer.encode(chunk))
        # print(f"Token count for current chunk: {token_count}")

        # If the current chunk does not exceed the max_token_length, add it to rows
        if token_count <= max_token_length:
            rows.append(chunk.strip())
        else:
            print("Chunk exceeds token limit; not adding.")

    # Filter out chunks below the minimum size
    return [c for c in rows if len(tokenizer.encode(c)) > min_chunk_size]

# Process each PDF file and store chunks with metadata
rows = []
for idx, row in enumerate(df.collect()):
    binary_data = row["content"]
    original_file_path = row["path"]  # Use the "path" column for file path
    filename = os.path.basename(original_file_path)  # Extract filename from path
    pdf_text = parse_pdf_content(binary_data) 
    if pdf_text:
        # Create chunks for each PDF's text based on tokens
        chunks = tokenize_and_split_text(pdf_text)
        total_chunks = len(chunks)
        for chunk_idx, chunk in enumerate(chunks):
            rows.append(Row(
                file_id=str(idx),
                chunk_index=chunk_idx,
                chunk_text=chunk,
                original_file_path=original_file_path,
                filename=filename,  # Add filename here
                total_chunks=total_chunks
            ))

# Convert rows to DataFrame and write to Delta table
chunks_df = spark.createDataFrame(rows, schema=schema)
chunks_df.write.mode("append").saveAsTable("pdf_text_chunks")


# COMMAND ----------

# MAGIC %md
# MAGIC Job Description embeddings

# COMMAND ----------

# DBTITLE 1,job description embeddings
import mlflow.deployments

# Load the job description text
job_description = "/Volumes/workspace/default/shortlist_ai/output/job_description.txt"
with open(job_description, "r") as file:
    job_description_text = file.read()

# Function to get embedding from model for a single input
def get_single_embedding(text: str):
    deploy_client = mlflow.deployments.get_deploy_client("databricks")
    response = deploy_client.predict(endpoint="databricks-gte-large-en", inputs={"input": text})
    return response.data[0]['embedding']  # Extract embedding directly

# Compute embedding for the job description
job_description_embeddings = get_single_embedding(job_description_text)

# Print or use the embedding
# print(job_description_embeddings)


# COMMAND ----------

# MAGIC %md
# MAGIC Vector search creation

# COMMAND ----------

# DBTITLE 1,create vector search
'''
from databricks.vector_search.client import VectorSearchClient

vsc = VectorSearchClient()
endpoint_name = "pdf_chunk_embeddings"

# Check if the endpoint already exists
existing_endpoints = vsc.list_endpoints()
endpoint_exists = endpoint_name in existing_endpoints

# Create the endpoint if it does not exist
if not endpoint_exists:
    endpoint = vsc.create_endpoint(name=endpoint_name, endpoint_type="STANDARD")
    print(f"Created vector search endpoint: {endpoint_name}")
else:
    print(f"Vector search endpoint '{endpoint_name}' already exists.")
'''

# COMMAND ----------

# MAGIC %md
# MAGIC Create vector index in the path
# MAGIC compute-> table "pdf_text_chunks" -> create-> vector search

# COMMAND ----------

# MAGIC %md
# MAGIC Similarity search

# COMMAND ----------

# Perform the similarity search using the generated query embedding
from databricks.vector_search.client import VectorSearchClient

client = VectorSearchClient()

endpoint = "pdf_chunk_embeddings"
vs_index_fullname = "workspace.default.vec_search_index"

results = client.get_index(endpoint, vs_index_fullname).similarity_search(
    query_vector=job_description_embeddings,
    columns=["filename","chunk_index","chunk_text"], 
    num_results=5
)
docs = results.get('result', {}).get('data_array', [])
# print(docs)

# COMMAND ----------

# DBTITLE 1,group chunks by filename and sort chunk by chunk index
# group chunks by filename and sort chunk by chunk index
from collections import defaultdict

grouped_chunks = defaultdict(list)

# Process and display the results
for doc in docs:
    filename = doc[0]        # Filename is the first item in each list
    chunk_index = int(doc[1]) # Chunk index is the second item
    chunk_text = doc[2]       # Chunk text is the third item
    # Group chunks by filename
    grouped_chunks[filename].append({
        "filename" : filename,
        "chunk_index": chunk_index,
        # "score": score,
        "content": chunk_text

    })

# Sort chunks in each file based on chunk_index
for filename in grouped_chunks:
    grouped_chunks[filename].sort(key=lambda x: x['chunk_index'])

# print(grouped_chunks)


# COMMAND ----------

# MAGIC %md
# MAGIC Ranking resumes
# MAGIC

# COMMAND ----------

# ranking resumes 
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

def rate_resume_with_llm(job_description, resume_content):
    w = WorkspaceClient()
    """Rate how well the resume matches the job description using OpenAI's GPT model."""
    prompt = f"""
    Based on the following job description:

    '{job_description}'

    Please review the following resume:

    '{resume_content}'

    Rate the resume on a scale of 1 to 10 for how well it matches the job description. Analyze the mandatory/key points from job description
    for more accurate rating.
    Provide a brief reason for the score, mentioning key points such as experience, skill match, etc.
    Additionally, extract the candidate's name and email from the resume. 
    Format the response as: Name: [name], Email: [email], Score: [score], Reason: [reason]
    """
    # Call the GPT model to get the rating
    response = w.serving_endpoints.query(
        name="databricks-mixtral-8x7b-instruct",
        messages=[ChatMessage(role=ChatMessageRole.USER, content=prompt)]
    )

    output = response.choices[0].message.content
    try:
            #print(output)
            name_str, email_str = output.split(", Email:")
            name = name_str.split(":")[1].strip()
            email, score_str = email_str.split(", Score:")
            score, reason_str = score_str.split(", Reason:")
            reason = reason_str.strip()

    except ValueError as e:
        print(f"Error parsing output: {output}")
        return 0, "Parsing error", "Unknown", "Unknown"  # Default return values on error

    return  name , email ,score, reason

def rank_resumes(resume_scores):
    """Rank resumes based on their score."""
    # Sort resume_scores list of tuples by score (which is at index 1)
    ranked_resumes = sorted(resume_scores, key=lambda x: float(x[3]), reverse=True)
    return ranked_resumes

def save_ranked_resumes(ranked_resumes, output_file):
   
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_file = os.path.join(output_dir, os.path.basename(output_file))
    
    with open(output_file, 'w') as file:
        file.write("Ranked Resumes:\n")
        for rank, (resume_filename, name, email, score, reason ) in enumerate(ranked_resumes, 1):
            file.write(f"{rank}. {resume_filename} - Name: {name}, Email: {email}, Score: {score}/10, Reason: {reason} \n")


resume_scores = []
for filename, chunks in grouped_chunks.items():
    # Concatenate all chunk contents to form a full resume
    resume_content = "\n".join(chunk['content'] for chunk in chunks)
    name , email ,score, reason = rate_resume_with_llm(job_description, resume_content)
    resume_scores.append((filename, name , email ,score, reason))
ranked_resumes = rank_resumes(resume_scores)
output_file = "/Volumes/workspace/default/shortlist_ai/output/ranked_resumes.txt"
save_ranked_resumes(ranked_resumes, output_file)
