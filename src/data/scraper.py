import requests
import re
import os
import time
from tqdm import tqdm

USER_AGENT = "STC-Transformer-DatasetBuilder/2.0 (adrian.badarau.stc@gmail.com; educational research tool)"

def make_request(url, params, max_retries=5):
    headers = {"User-Agent": USER_AGENT}
    delay = 1.0  # start with 1 second backoff
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, params=params, headers=headers)
            
            # Handle rate limiting explicitly
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        sleep_time = int(retry_after)
                    except ValueError:
                        sleep_time = 5
                else:
                    sleep_time = delay
                print(f"\n[429] Rate limited on {params.get('titles') or params.get('cmtitle')}. Sleeping {sleep_time}s before retry (attempt {attempt+1}/{max_retries})...")
                time.sleep(sleep_time)
                delay *= 2
                continue
                
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                time.sleep(delay)
                delay *= 2
                continue
            print(f"HTTP Error: {e}")
            time.sleep(delay)
            delay *= 2
        except Exception as e:
            print(f"Request Exception: {e}")
            time.sleep(delay)
            delay *= 2
            
    return None

def get_category_members(category_name, limit=50):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"Category:{category_name}",
        "cmlimit": limit,
        "format": "json"
    }
    
    data = make_request(url, params)
    if not data:
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
    
    data = make_request(url, params)
    if not data:
        return ""
        
    pages = data.get("query", {}).get("pages", {})
    for page_id, page_info in pages.items():
        if "extract" in page_info:
            return page_info["extract"]
    return ""

def is_relevant_title(title):
    title_lower = title.lower()
    # Standard Wikipedia namespaces
    namespaces = ["category:", "template:", "file:", "talk:", "user:", "wikipedia:", "portal:", "draft:", "help:", "mediawiki:"]
    if any(ns in title_lower for ns in namespaces):
        return False
    
    # Administrative/List pages
    list_patterns = ["list of", "index of", "outline of", "glossary of", "timeline of"]
    if any(pat in title_lower for pat in list_patterns):
        return False
        
    # Irrelevant administrative or non-technical topics
    noise_patterns = [
        "regulation", "accidents", "disaster", "strike", "museum", 
        "association", "society", "legislation", "environmental impact"
    ]
    if any(pat in title_lower for pat in noise_patterns):
        return False
        
    return True

def format_text_as_qa(title, text):
    # Remove references/sources/external links sections at the end of the text
    end_headers = [
        "== see also ==", 
        "== references ==", 
        "== further reading ==", 
        "== external links ==", 
        "== sources ==",
        "== bibliography ==",
        "== notes =="
    ]
    
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        if line.strip().lower() in end_headers:
            break
        cleaned_lines.append(line)
        
    text = '\n'.join(cleaned_lines)
    
    # Parse text into sections matching == Section Name ==
    pattern = r'^(={2,6})\s*(.*?)\s*\1\s*$'
    
    sections = []
    current_section_title = "Introduction"
    current_section_lines = []
    
    for line in text.split('\n'):
        match = re.match(pattern, line.strip())
        if match:
            # Save previous section if it has content
            content = " ".join([l.strip() for l in current_section_lines if l.strip()])
            if content:
                sections.append((current_section_title, content))
            
            # Start new section
            current_section_title = match.group(2).strip()
            current_section_lines = []
        else:
            current_section_lines.append(line)
            
    # Save the last section
    content = " ".join([l.strip() for l in current_section_lines if l.strip()])
    if content:
        sections.append((current_section_title, content))
        
    # Format each section into Q&A style
    formatted_qa = []
    for sec_title, sec_content in sections:
        sec_content = re.sub(r'\s+', ' ', sec_content).strip()
        if not sec_content:
            continue
            
        if sec_title.lower() == "introduction":
            prompt = f"Question: What is {title}?"
        else:
            prompt = f"Question: In the context of {title}, tell me about {sec_title}."
        
        response = f"Answer: {sec_content}"
        formatted_qa.append(f"{prompt}\n{response}")
        
    return "\n\n".join(formatted_qa)

def build_corpus(categories, output_file, max_pages_per_cat=50):
    print("Building expanded corpus from Wikipedia...")
    
    curated_articles = [
        # Stone & Tools
        "Stone tool", "Flintknapping", "Tool stone", "Oldowan", "Acheulean", 
        "Hand axe", "Axe", "Adze", "Wedge (woodworking)",
        # Fire & Charcoal
        "Firemaking", "Bow drill", "Fire drill", "Charcoal", "Charcoal pile", 
        "Charcoal burner", "Pyrolysis", "Biochar",
        # Smelting & Metals
        "Smelting", "Bloomery", "Blast furnace", "Bog iron", "Iron ore", 
        "Wrought iron", "Cast iron", "Steel", "High-carbon steel", 
        "Ferrous metallurgy", "Extractive metallurgy", "Copper extraction",
        # Forge Craft
        "Blacksmith", "Forging", "Anvil", "Forge", "Bellows", "Tuyere", "Quenching",
        # Other Sourcing
        "Wood", "Lumber", "Woodworking", "Clay", "Pottery", "Kiln", "Lime kiln", 
        "Bushcraft", "Survival skills", "Wattle and daub", "Rope", "Pine tar"
    ]
    
    unique_titles = set(curated_articles)
    
    # Add category member pages
    for category in categories:
        print(f"Fetching page list for category: {category}")
        pages = get_category_members(category, limit=max_pages_per_cat)
        
        filtered_count = 0
        for title in pages:
            if is_relevant_title(title):
                unique_titles.add(title)
            else:
                filtered_count += 1
        if filtered_count > 0:
            print(f"  Filtered out {filtered_count} administrative/noise pages from {category}.")
            
        time.sleep(0.5)  # Be polite between category calls
            
    print(f"Total unique articles to fetch: {len(unique_titles)}")
    all_text = []
    
    for title in tqdm(sorted(list(unique_titles))):
        text = extract_text_from_wikipedia(title)
        if text:
            formatted_qa = format_text_as_qa(title, text)
            if formatted_qa:
                all_text.append(formatted_qa)
        time.sleep(0.5) # Polite delay (0.5s) to prevent 429
            
    final_text = "\n\n".join(all_text)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(final_text)
        
    print(f"Corpus saved to {output_file} ({len(final_text) / 1024 / 1024:.2f} MB)")

if __name__ == "__main__":
    categories = [
        "Mining", "Quarrying", "Clay", "Iron ores", "Lithics", "Woodworking", 
        "Forestry", "Charcoal", "Smelting", "Charcoal ovens", "Survival skills",
        "Swords", "Metallurgy", "Blacksmithing"
    ]
    build_corpus(categories, "stc_training_data.txt", max_pages_per_cat=60)
