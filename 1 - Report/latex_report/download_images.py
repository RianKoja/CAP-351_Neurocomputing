# Function that receives a search query and returns google image results for that query.
#
# Usage: python download_images.py <search_query>
#
# Dependencies:
# - Python 3
# - BeautifulSoup (install using pip)
# - Selenium (install using pip)
#
#
#

import sys
import os
import re
import time
import urllib.request
from bs4 import BeautifulSoup
from selenium import webdriverdef


def get_images(search_query):
    # Create a new Firefox session
    driver = webdriver.Firefox()
    driver.implicitly_wait(30)
    driver.get("https://www.google.com")
    # Navigate to the application home page
    driver.find_element_by_id("gbqfq").send_keys(search_query + " " + "images")
    driver.find_element_by_id("gbqfb").click()
    # Wait for page to load
    time.sleep(3)
    # Pass the source of the page to BeautifulSoup
    source = driver.page_source
    soup = BeautifulSoup(source, "lxml")
    # Find all the links on the page
    images = soup.find_all("img")
    # Create a new directory to store results
    if not os.path.exists(search_query):
        os.makedirs(search_query)
    # Loop through the images and save them to the directory
    for i in range(len(images)):
        image_path = images[i]["src"]
        try:
            urllib.request.urlretrieve(image_path, search_query + "/" + str(i) + ".jpg")
        except Exception:
            print("Unable to download: " + image_path)
    driver.close()

    # unit test
    get_images("cat")
