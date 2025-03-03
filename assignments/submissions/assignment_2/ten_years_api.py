import requests
import pandas as pd
from scipy import stats

import requests
import csv
import os

# Example journal ISSNs
journals = [
    {"name": "Academy of Management Journal", "issn": "0001-4273"},
    {"name": "Strategic Management Journal", "issn": "0143-2095"},
    {"name": "Journal of Business Venturing", "issn": "0883-9026"}
]

results = []

# Create directory for output files if it doesn't exist
output_directory = 'assignment_2'
os.makedirs(output_directory, exist_ok=True)

# Fetch articles from each journal
for journal in journals:
    issn = journal["issn"]
    url = f"https://api.crossref.org/journals/{issn}/works?filter=from-pub-date:2015-01-01"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        for item in data['message']['items']:
            results.append({
                'DOI': item.get('DOI', ''),
                'Title': item.get('title', [''])[0],
                'References': item.get('reference-count', 0),
                'Authors': len(item.get('author', [])),
                'Citations': item.get('is-referenced-by-count', 0),
                'URL': item.get('URL', ''),
                'Journal': journal["name"],
                'Abstract': item.get('abstract', ''),
                'Publication Year': item.get('published-print', {}).get('date-parts', [[None]])[0][0]
            })
    else:
        print(f"Error fetching data for journal {journal['name']} (ISSN: {issn}): {response.status_code} - {response.text}")

# Save results to CSV
csv_file_path = os.path.join(output_directory, "ten_years_api.csv")
with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['DOI', 'Title', 'References', 'Authors', 'Citations', 'URL', 'Journal', 'Abstract', 'Publication Year']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for result in results:
        writer.writerow(result)

print(f"Data saved to {csv_file_path}")