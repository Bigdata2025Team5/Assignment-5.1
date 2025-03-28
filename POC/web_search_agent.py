import requests
import os
from dotenv import load_dotenv

load_dotenv()

def fetch_web_data(query: str):
    try:
        api_key = os.getenv("WEB_SEARCH_API_KEY")
        
        print(f"Using API Key: {api_key}")  # Debugging line
        
        # Perform the web search using the API
        response = requests.get(f"https://serpapi.com/search?q={query}+NVIDIA&api_key={api_key}")
        
        if response.status_code != 200:
            print(f"Failed to fetch data from SerpAPI. Status code: {response.status_code}")
            return {"insights": "Failed to fetch web insights."}
        
        data = response.json()
        
        # Check if organic results are present
        if "organic_results" in data and data["organic_results"]:
            top_result = data["organic_results"][0]
            insight = top_result.get("snippet", "No relevant insight found.")
            links = [result.get("link", "No link found.") for result in data["organic_results"]]
        else:
            insight = "No relevant insights found."
            links = []
        
        print(f"Fetched web data: {insight}")
        return {"insights": insight, "links": links}
    
    except Exception as e:
        # Log the error if the web request fails
        print(f"Error occurred while fetching web insights: {str(e)}")
        return {"insights": "Failed to fetch web insights due to an error.", "links": []}
