import json
import csv

# Specify input and output file paths
input_file_path = 'data/datasets/huffpost-news.json'  # replace with the path to your JSONL file
output_file_path = 'data/datasets/huffpost-news.csv'  # replace with the desired path for your output CSV

# Open the input JSONL file and output CSV file
with open(input_file_path, 'r') as jsonl_file, open(output_file_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file, delimiter=',')
    
    # Write the header row
    csv_writer.writerow(['category', 'headline'])
    
    # Process each line in the JSONL file
    for line in jsonl_file:
        # Parse the JSON object
        json_obj = json.loads(line)
        
        # Extract 'category' and 'headline' and write to CSV
        csv_writer.writerow([json_obj.get('category', ''), json_obj.get('headline', '')])

print("Conversion completed successfully!")
