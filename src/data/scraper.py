import requests
import re
from bs4 import BeautifulSoup
import os
import time
from tqdm import tqdm

def get_category_members(category_name, limit=50):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category_name}",
        "cmlimit": limit,
        "format": "json"
    }
    headers = {"User-Agent": "STC-Transformer-Project/1.0 (contact@example.com)"}
    response = requests.get(url, params=params, headers=headers)
    
    try:
        data = response.json()
    except requests.exceptions.JSONDecodeError:
        print(f"Error decoding JSON. Response text: {response.text[:200]}")
        return []

    members = data.get("query", {}).get("categorymembers", [])
    
    # Filter for regular pages (ns=0)
    pages = [m["title"] for m in members if m["ns"] == 0]
    return pages

def extract_text_from_wikipedia(title):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "titles": title,
        "explaintext": True,
        "format": "json"
    }
    headers = {"User-Agent": "STC-Transformer-Project/1.0 (contact@example.com)"}
    response = requests.get(url, params=params, headers=headers)
    
    try:
        data = response.json()
    except requests.exceptions.JSONDecodeError:
        print(f"Error decoding JSON for {title}. Response text: {response.text[:200]}")
        return ""

    pages = data.get("query", {}).get("pages", {})
    for page_id, page_info in pages.items():
        if "extract" in page_info:
            return page_info["extract"]
    return ""

def clean_text(text):
    # Remove extra whitespace and newlines
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'\s+', ' ', text)
    # Remove section headers like == See also ==
    text = re.sub(r'==.*?==', '', text)
    return text.strip()

def build_corpus(categories, output_file, max_pages_per_cat=50):
    print("Building corpus from Wikipedia...")
    all_text = []
    
    for category in categories:
        print(f"Fetching pages for category: {category}")
        pages = get_category_members(category, limit=max_pages_per_cat)
        
        for title in tqdm(pages):
            text = extract_text_from_wikipedia(title)
            cleaned = clean_text(text)
            if cleaned:
                all_text.append(cleaned)
            time.sleep(0.1) # Be polite to Wikipedia API
            
    final_text = "\n".join(all_text)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(final_text)
        
    print(f"Corpus saved to {output_file} ({len(final_text) / 1024 / 1024:.2f} MB)")

if __name__ == "__main__":
    categories = ["Swords", "Metallurgy", "Blacksmithing"]
    build_corpus(categories, "stc_training_data.txt")
