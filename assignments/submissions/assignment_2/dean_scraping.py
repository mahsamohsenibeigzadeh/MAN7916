import requests
from bs4 import BeautifulSoup
import csv

import requests
from bs4 import BeautifulSoup

url = "https://pauljarley.wordpress.com/"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

posts = []
for post in soup.find_all('article'):
    title_tag = post.find('h1')  # Change this if needed
    if title_tag:
        title = title_tag.text
        date_tag = post.find('time')
        date = date_tag['datetime'] if date_tag else 'No date found'
        post_url = post.find('a')['href'] if post.find('a') else 'No URL found'
        posts.append((title, date, post_url))
    else:
        print("Post structure has changed or title is missing.")

# Print collected posts
for title, date, post_url in posts:
    print(f"Title: {title}, Date: {date}, URL: {post_url}")

post_data = []
for title, date, post_url in posts:
    post_response = requests.get(post_url)
    post_soup = BeautifulSoup(post_response.text, 'html.parser')


    post_content_div = post_soup.find('div', class_='entry-content')  # Change this to the right class name
    text = post_content_div.text.strip() if post_content_div else 'Post content not found.'


    comments = []
    for comment in post_soup.find_all('div', class_='comment'):
        author_tag = comment.find('span', class_='comment-author')
        author = author_tag.text if author_tag else 'Anonymous'
        comment_text_tag = comment.find('p')
        comment_text = comment_text_tag.text if comment_text_tag else 'No text available.'
        comments.append((author, comment_text))

    post_data.append((title, date, post_url, text, comments))

# Print collected post data
for post in post_data:
    print(post)

for post in post_data:
    print(post)  # This will show the structure of each post

import os
import csv

# Create the directory if it doesn't exist
if not os.path.exists('assignment_2'):
    os.makedirs('assignment_2')

# Check if post_data is populated
if post_data:
    with open('assignment_2/dean_scraping.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Title', 'Date', 'URL', 'Post Text', 'Comments'])

        for title, date, post_url, text, comments in post_data:
            comment_texts = "; ".join([f'{author}: {comment_text}' for author, comment_text in comments])
            writer.writerow([title, date, post_url, text, comment_texts])

    print("Data written to dean_scraping.csv successfully.")
else:
    print("No post data collected.")

