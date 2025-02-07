import os
import time
import requests
from urllib.parse import urlparse
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# Create output directory for downloaded images
DOWNLOAD_FOLDER = "downloaded_images"
if not os.path.exists(DOWNLOAD_FOLDER):
    os.makedirs(DOWNLOAD_FOLDER)

# Configure Selenium to work with headless Chrome
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--window-size=1920,1080")
chrome_options.add_argument(
    "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/70.0.3538.77 Safari/537.36"
)

# If necessary, specify the path to chromedriver:
# driver = webdriver.Chrome(executable_path="path_to_chromedriver", options=chrome_options)
driver = webdriver.Chrome(options=chrome_options)

# Define the search query (no hardcoding the full URL)
query = "pencil sketch drawing"
query_formatted = query.replace(" ", "+")
url = f"https://www.google.com/search?q={query_formatted}&tbm=isch"

print("Searching for:", query)
driver.get(url)
time.sleep(2)  # Initial wait for the page to load

# Use a set to avoid duplicate URLs
image_urls = set()
max_scrolls = 20  # Maximum number of scroll iterations
scroll_pause_time = 2
scroll_count = 0

while scroll_count < max_scrolls:
    # Scroll to the bottom to load more images
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(scroll_pause_time)

    # Attempt to click the "Show more results" button if it exists
    try:
        show_more_button = driver.find_element(By.CSS_SELECTOR, ".mye4qd")
        if show_more_button:
            show_more_button.click()
            time.sleep(2)
    except Exception:
        # If the button isn't found, continue scrolling
        pass

    # Collect image elements on the current view
    images = driver.find_elements(By.TAG_NAME, "img")
    for img in images:
        src = img.get_attribute("src")
        if src and src.startswith("http"):
            image_urls.add(src)

    scroll_count += 1
    print(f"Scroll count: {scroll_count}, Total images collected: {len(image_urls)}")
    
    # Break early if we collected enough URLs
    if len(image_urls) >= 500:
        break

print(f"Collected {len(image_urls)} valid image URLs.")

# Download images, limiting to a maximum of 500 images
max_images = 500
downloaded = 0
for idx, img_url in enumerate(image_urls):
    if downloaded >= max_images:
        break
    try:
        response = requests.get(img_url, stream=True, timeout=10)
        if response.status_code == 200:
            parsed_url = urlparse(img_url)
            filename = os.path.basename(parsed_url.path)
            # If filename is missing or lacks extension, supply a default name with .jpg extension
            if not filename or not os.path.splitext(filename)[1]:
                filename = f"img_{downloaded}.jpg"
            else:
                filename = filename.split("?")[0]  # Remove query parameters if any
            # Prefix with index to ensure uniqueness
            filename = f"{downloaded}_{filename}"
            file_path = os.path.join(DOWNLOAD_FOLDER, filename)
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"Downloaded image {downloaded}: {file_path}")
            downloaded += 1
    except Exception as e:
        print(f"Failed to download image {idx} from {img_url}. Error: {e}")

driver.quit()
print("Finished downloading images.")