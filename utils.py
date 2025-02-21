import re
import wikipedia # type: ignore
from langchain_groq import ChatGroq  # type: ignore #*********************************************************************
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage , RemoveMessage # type: ignore
from tavily import TavilyClient
import os
from dotenv import load_dotenv # type: ignore
from langchain_community.tools import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchRun
from spellchecker import SpellChecker
import re

spell = SpellChecker()


def remove_think_content(text):
    think_pattern = r'<think>(.*?)</think>'
    cleaned_text = re.sub(think_pattern, '', text, flags=re.DOTALL)
    return cleaned_text

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
def clean_pdf_text(content):

    content = re.sub(r'\s+', ' ', content)

    content = re.sub(r'\x0c', '', content)

    content = re.sub(r'[^\x00-\x7F]+', '', content)

    content = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', content)

    words = content.split()

    corrected = [spell.correction(word) if word in spell else word for word in words]

    content = ' '.join([line.strip() for line in corrected.split('\n') if line.strip()])

    return content


def search_for_page(query: str) -> list:
    wikipedia.set_lang('en')  
    print(bcolors.OKCYAN + f'Searching for pages' + bcolors.ENDC)
    results = wikipedia.search(query)[:2]
    print(bcolors.OKCYAN + f'Found pages {results}' + bcolors.ENDC)
    return results


def get_wiki_content(queries: list) -> list:
    res = []
    for query in queries:
        try:
            res.append(wikipedia.page(query))
        except wikipedia.exceptions.PageError:
            print(bcolors.FAIL + f"Page not found for query: {query}" + bcolors.ENDC)
            continue  
        except wikipedia.exceptions.DisambiguationError as e:
            print(bcolors.FAIL + f"Disambiguation error for query: {query}. Options: {e.options}" + bcolors.ENDC)
            continue  
    return res 

    
def get_res(query , model_name):# This version uses wikipedia only
    
    print(bcolors.WARNING + f'Using the wikipedia tool:...\n' + bcolors.ENDC)
    model = ChatGroq(model_name=model_name, temperature=0)
    search_query = model.invoke([SystemMessage(content=f'''
    Convert the user query into search keyword in wikipedia,
    ensure correct understanding of the query before the conversion,
    it is in the context of syrian history and culure.
    return only and only the keyword: {query}''')]).content
    
    pages = search_for_page(query)
    if not pages:
        return "No Wikipedia pages found for this query."

    content = get_wiki_content(pages)
    if not content:
        return "No results found."
    
    formatted_results = []
    for page in content:
        formatted_results.append(
        f"Title: {page.title}\n"
        f"Summary: {page.summary[:800]}...\n"
        f"URL: {page.url}\n"
        )
    results = '\n\n'.join(formatted_results)
    print(bcolors.WARNING + f'Returned Results from wikipedia' + bcolors.ENDC)
    return results

def get_tavily_search(query):  

    print(bcolors.WARNING + 'Using the Tavily search tool...\n' + bcolors.ENDC)

    try:
        load_dotenv()
        api_key = os.getenv("TAVILY_API_KEY")

        if not api_key:
            print(bcolors.WARNING + "Warning: TAVILY_API_KEY is not set. Returning empty response.\n" + bcolors.ENDC)
            return ""

        os.environ["TAVILY_API_KEY"] = api_key

        tool = TavilySearchResults(
            max_results=5,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=True,
            include_images=False,
        )

        res = tool.invoke({"query": query})
        content = ""

        for result in res:
            content += f"\n Source: {result.get('url', 'N/A')} - Result: {result.get('content', 'No content')}\n"

        print(content)
        print(bcolors.WARNING + "Returned results from Tavily" + bcolors.ENDC)
        return content

    except Exception as e:
        print(bcolors.WARNING + f"Error occurred in Tavily search: {e}. Returning empty response.\n" + bcolors.ENDC)
        return ""

def get_duckduckgo_search(query):

    try:
        search = DuckDuckGoSearchRun()
        return search.invoke(query)
    except Exception as e:
        print(bcolors.WARNING + f"Error occurred in DuckDuckGo search: {e}. Returning empty response.\n" + bcolors.ENDC)
        return ""