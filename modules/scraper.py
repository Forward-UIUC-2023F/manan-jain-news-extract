import requests

from pyppeteer import launch
import asyncio
import nest_asyncio
nest_asyncio.apply()

from bs4 import BeautifulSoup
from bs4.element import Tag


def fetch_html(url: str, waitUntil: str='networkidle0', setUserAgent: bool=False) -> str:
    """Fetches the raw text html of a webpage given a url. Works with dynamic pages

    Args:
        url (str): URL of website to fetch html from
        waitUntil (str, optional): What condition to wait for to assume a full page load. Defaults to 'networkidle0'. (means 0 network activity for 500 ms)
        
    Returns:
        str: html of a webpage in plain text
    """
    if waitUntil == 'static':
            page = requests.get(url, {})
            return page.content
        
    async def main():
        browser = await launch()
        page = await browser.newPage()
        if setUserAgent:
            await page.setUserAgent("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.75 Safari/537.36")
        await page.goto(url, {'waitUntil': waitUntil})
        page_content = await page.content()
        await browser.close()
        return page_content
        
    loop = asyncio.get_event_loop()
    html = loop.run_until_complete(main())
    return html

def clean_html(soup: BeautifulSoup) -> BeautifulSoup:
    """Clean up html by removing unneccesary tags including: ['script', 'style', 'meta', 'link', 'svg']

    Args:
        soup (BeautifulSoup): soup object of the html page to clean up
    
    Returns: 
        BeautifulSoup: soup object representing condensed html without ['script', 'style', 'meta', 'link', 'svg'] tags
    """
    for tag in ['script', 'style', 'meta', 'link', 'svg']:
        for element in soup.find_all(tag):
            element.decompose()
            
    return soup
        
def merge_attributes(attrs1: dict, attrs2: dict):
    """Merge two dictionaries of attributes.

    Args:
        attrs1 (dict): dictionary of attribute: values for Tag 1
        attrs2 (dict): dictionary of attribute: values for Tag 2

    Returns:
        dict: dictionary of attribute: values pair for the merged html tags.
    """
    merged = attrs1.copy()
    for key, value in attrs2.items():
        if key in merged:
            # Concatenate values for the same attribute
            merged[key] = f"{merged[key]} {value}"
        else:
            merged[key] = value
    return merged

def squash_nested_divs(soup: BeautifulSoup) -> BeautifulSoup:
    """Squash consecutively nested divs into a single div and merge attributes.

    Args:
        soup (BeautifulSoup): soup object for which consecutive div tags must be squashed

    Returns:
        BeautifulSoup: soup object with squashed and merged divs
    """
    # soup = BeautifulSoup(html, 'html.parser')

    divs = soup.find_all('div')
    for div in divs:
        while div.find('div'):  # Check for a nested div
            nested_div = div.find('div')
            div.attrs = merge_attributes(div.attrs, nested_div.attrs)
            nested_div.unwrap()

    return soup

def extract_tags(soup: BeautifulSoup) -> "list[str]":
    """Extract HTML contents from each <a>, <article>, and <section> tag including the <a>, <article>, or <article> tag itself.

    Args:
        soup (BeautifulSoup): soup object from which tags must be extracted

    Returns:
        List[str]: List of all the extracted tags as plaintext html
    """
    tags = soup.find_all(['a', 'article', 'section'])
    # Convert each <a> tag back to a string
    tags_html = [str(tag) for tag in tags]

    return tags_html