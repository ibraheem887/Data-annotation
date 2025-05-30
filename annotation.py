import aiohttp
import asyncio
import os
import pandas as pd
import google.generativeai as genai
import time
import csv
import random
import math
from bs4 import BeautifulSoup
import validators

# ✅ Set API key
GEMINI_API_KEY = "AIzaSyCIu5m1UXc0IhYYHrNE1wBsgfWEoeAf5ig"
genai.configure(api_key=GEMINI_API_KEY)

# 🔹 CSV file paths
input_csv = "E:/Data science/DS Web scrapping/output.csv"
output_csv = "E:/Data science/DS Web scrapping/annotated_papers.csv"

# 🔹 Define valid categories
CATEGORIES = ["Deep Learning", "Reinforcement Learning", "Optimization", "Graph Neural Networks"]

# ✅ Load CSV file
df = pd.read_csv(input_csv)

# ✅ Track already processed papers
existing_titles = set()
try:
    with open(output_csv, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader, None)
        if header:
            for row in reader:
                existing_titles.add(row[1])  # Title is in the 2nd column
except FileNotFoundError:
    pass  # No previous results, start fresh

async def fetch_abstract_async(session, paper_link):
    """Fetches the abstract from the second <p> tag inside the paper link."""
    if not validators.url(paper_link):
        return "No abstract found."

    for attempt in range(5):
        try:
            async with session.get(paper_link, timeout=60) as response:
                if response.status == 200:
                    text = await response.text()
                    soup = BeautifulSoup(text, "html.parser")
                    abstract_tags = soup.body.find_all("p")
                    return abstract_tags[2].text.strip() if len(abstract_tags) > 1 else "No abstract found."
        except Exception:
            pass

    return "No abstract found."

async def fetch_with_semaphore(semaphore, session, paper_link):
    """Ensures that abstracts are fetched safely with rate limits."""
    async with semaphore:
        return await fetch_abstract_async(session, paper_link)

async def fetch_all_abstracts(paper_links, max_concurrent_requests=5):
    """Fetches abstracts for a batch of paper links."""
    semaphore = asyncio.Semaphore(max_concurrent_requests)
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_with_semaphore(semaphore, session, link) for link in paper_links]
        return await asyncio.gather(*tasks)

def classify_papers_batch(papers):
    """Classifies a batch of research papers using Google Gemini."""
    prompt = "You are an AI classifier. Categorize each paper into ONE of the following categories ONLY:\n"
    for i, category in enumerate(CATEGORIES, 1):
        prompt += f"{i}. {category}\n"

    prompt += "\nHere are the research papers:\n"
    for i, (title, abstract) in enumerate(papers, 1):
        prompt += f"\nPaper {i}:\nTitle: {title}\nAbstract: {abstract}\n"

    prompt += "\nRespond with a numbered list of categories for each paper. Only return category names from the list above."

    for attempt in range(5):
        try:
            model = genai.GenerativeModel("gemini-pro")
            response = model.generate_content(prompt)
            if not response.text:
                raise ValueError("Empty response from API")

            categories = response.text.strip().split("\n")
            valid_categories = [next((cat for cat in CATEGORIES if cat in line), "Unknown") for line in categories]
            while len(valid_categories) < len(papers):
                valid_categories.append("Unknown")
            while len(valid_categories) > len(papers):
                valid_categories.pop()
            return valid_categories
        except Exception:
            pass
        time.sleep(5 * (2 ** attempt))

    return ["Unknown"] * len(papers)

async def process_papers():
    """Processes all research papers in batches."""
    BATCH_SIZE = 10
    with open(output_csv, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not existing_titles:
            writer.writerow(["Year", "Title", "Authors", "Paper Link", "Abstract", "Category"])
        total_batches = math.ceil(len(df) / BATCH_SIZE)
        for batch_num in range(total_batches):
            batch_start = batch_num * BATCH_SIZE
            batch_end = min(batch_start + BATCH_SIZE, len(df))
            batch_papers, batch_links, batch_meta = [], [], []
            for index in range(batch_start, batch_end):
                year, title, authors, paper_link = df.iloc[index]["Year"], df.iloc[index]["Title"], df.iloc[index]["Authors"], df.iloc[index]["Paper Link"]
                if title in existing_titles:
                    continue
                batch_links.append(paper_link)
                batch_papers.append((title, None))
                batch_meta.append((year, title, authors, paper_link))
            if not batch_papers:
                continue
            print(f"\n🔍 Fetching abstracts for batch {batch_num + 1}/{total_batches}...")
            abstracts = await fetch_all_abstracts(batch_links)
            batch_papers = [(title, abstract) for (title, _), abstract in zip(batch_papers, abstracts)]
            print("✅ Abstracts fetched. Classifying papers...")
            batch_categories = classify_papers_batch(batch_papers)
            print("✅ Classification complete. Saving results...")
            for i in range(len(batch_papers)):
                writer.writerow([batch_meta[i][0], batch_papers[i][0], batch_meta[i][2], batch_meta[i][3], batch_papers[i][1], batch_categories[i]])
                file.flush()
            print(f"✅ Batch {batch_num + 1} processed successfully!")
            time.sleep(random.uniform(40, 60))
    print(f"\n✅ Process complete! Results saved to: {output_csv}")

asyncio.run(process_papers())
