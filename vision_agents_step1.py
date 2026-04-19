import os
from io import BytesIO
from time import sleep
from urllib.parse import quote_plus

import helium
import requests
from dotenv import load_dotenv
from PIL import Image
from smolagents import ActionStep, CodeAgent, DuckDuckGoSearchTool, InferenceClientModel, OpenAIModel, tool
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

image_urls = [
    "https://upload.wikimedia.org/wikipedia/commons/e/e8/The_Joker_at_Wax_Museum_Plus.jpg",  # Joker image
    "https://upload.wikimedia.org/wikipedia/en/9/98/Joker_%28DC_Comics_character%29.jpg",  # Joker image
]

load_dotenv()

MODEL_CHOICE = os.getenv("VISION_MODEL", "hf-qwen3-vl-8b")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "800"))


def require_driver():
    driver = helium.get_driver()
    if driver is None:
        raise RuntimeError("No active browser driver. Start a browser session before using browser tools.")
    return driver


def wait_for_page_ready(expected_url_fragment: str | None = None, timeout: int = 10):
    driver = require_driver()
    WebDriverWait(driver, timeout).until(lambda d: d.execute_script("return document.readyState") == "complete")
    WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
    if expected_url_fragment:
        WebDriverWait(driver, timeout).until(lambda d: expected_url_fragment in d.current_url)
    return driver


def build_model(model_choice: str):
    if model_choice == "openrouter-gpt-4o":
        openrouter_api_key = os.getenv("OPEN_ROUTER_API_KEY")
        if not openrouter_api_key:
            raise ValueError("Set OPEN_ROUTER_API_KEY to use openrouter-gpt-4o.")

        return OpenAIModel(
            model_id="openai/gpt-4o",
            api_base="https://openrouter.ai/api/v1",
            api_key=openrouter_api_key,
            max_tokens=MAX_TOKENS,
        )

    if model_choice in {
        "hf-qwen2.5-vl-32b",
        "hf-qwen2.5-vl-7b",
        "hf-qwen3-vl-8b",
    }:
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not hf_token:
            raise ValueError(
                "Set HF_TOKEN or HUGGINGFACEHUB_API_TOKEN to use the Hugging Face Qwen VLM option."
            )

        model_kwargs = {
            "model_id": "Qwen/Qwen3-VL-8B-Instruct",
            "token": hf_token,
            "max_tokens": MAX_TOKENS,
        }

        hf_provider = os.getenv("HF_PROVIDER")
        if hf_provider:
            model_kwargs["provider"] = hf_provider

        return InferenceClientModel(**model_kwargs)

    raise ValueError(
        "Unsupported VISION_MODEL. Use 'openrouter-gpt-4o' or 'hf-qwen3-vl-8b'."
    )

images = []
for url in image_urls:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content)).convert("RGB")
    images.append(image)

model = build_model(MODEL_CHOICE)


@tool
def open_browser(url: str = "https://www.google.com") -> str:
    """
    Opens Chrome and navigates to a URL.
    Args:
        url: The URL to open in the browser.
    """
    driver = helium.get_driver()
    if driver is None:
        helium.start_chrome(url)
    else:
        helium.go_to(url)
    return f"Opened browser at {url}"


@tool
def search_in_browser(query: str) -> str:
    """
    Opens DuckDuckGo search results for a query in the browser.
    Args:
        query: The search query to run in DuckDuckGo.
    """
    url = f"https://duckduckgo.com/?q={quote_plus(query)}"
    driver = helium.get_driver()
    if driver is None:
        helium.start_chrome(url)
    else:
        helium.go_to(url)
    return f"Opened DuckDuckGo search results for: {query}"


@tool
def open_wikipedia_page(title: str) -> str:
    """
    Opens the English Wikipedia page for a character or topic.
    Args:
        title: The Wikipedia page title, for example 'Wonder Woman' or 'Spider-Man'.
    """
    url = f"https://en.wikipedia.org/wiki/{quote_plus(title.replace(' ', '_'))}"
    driver = helium.get_driver()
    if driver is None:
        helium.start_chrome(url)
    else:
        helium.go_to(url)
    return f"Opened Wikipedia page: {url}"


@tool
def open_wikipedia_and_screenshot(title: str, filename: str) -> str:
    """
    Opens an English Wikipedia page and immediately saves a screenshot after the page loads.
    Args:
        title: The Wikipedia page title, for example 'Wonder Woman' or 'Spider-Man'.
        filename: The PNG filename to save in the current project directory.
    """
    url = f"https://en.wikipedia.org/wiki/{quote_plus(title.replace(' ', '_'))}"
    driver = helium.get_driver()
    if driver is None:
        helium.start_chrome(url)
    else:
        helium.go_to(url)

    wait_for_page_ready("/wiki/")

    if not filename.lower().endswith(".png"):
        filename = f"{filename}.png"
    output_path = os.path.abspath(filename)
    require_driver().save_screenshot(output_path)
    return f"Opened Wikipedia page and saved screenshot to {output_path}"


@tool
def search_item_ctrl_f(text: str, nth_result: int = 1) -> str:
    """
    Searches for text on the current page via Ctrl + F and jumps to the nth occurrence.
    Args:
        text: The text to search for
        nth_result: Which occurrence to jump to (default: 1)
    """
    driver = require_driver()
    elements = driver.find_elements(By.XPATH, f"//*[contains(text(), '{text}')]")
    if nth_result > len(elements):
        raise Exception(f"Match n°{nth_result} not found (only {len(elements)} matches found)")
    result = f"Found {len(elements)} matches for '{text}'."
    elem = elements[nth_result - 1]
    driver.execute_script("arguments[0].scrollIntoView(true);", elem)
    result += f"Focused on element {nth_result} of {len(elements)}"
    return result


@tool
def go_back() -> None:
    """Goes back to previous page."""
    driver = require_driver()
    driver.back()


@tool
def close_popups() -> str:
    """
    Closes any visible modal or pop-up on the page. Use this to dismiss pop-up windows! This does not work on cookie consent banners.
    """
    driver = require_driver()
    webdriver.ActionChains(driver).send_keys(Keys.ESCAPE).perform()
    return "Sent Escape to close visible popups."


@tool
def take_browser_screenshot(filename: str = "browser_screenshot.png") -> str:
    """
    Saves a screenshot of the current browser page to a PNG file.
    Args:
        filename: The PNG filename to save in the current project directory.
    """
    driver = require_driver()
    if not filename.lower().endswith(".png"):
        filename = f"{filename}.png"
    output_path = os.path.abspath(filename)
    driver.save_screenshot(output_path)
    return f"Saved browser screenshot to {output_path}"


def save_screenshot(step_log: ActionStep, agent: CodeAgent) -> None:
    sleep(1.0)
    driver = helium.get_driver()
    if driver is None:
        return

    current_step = step_log.step_number
    if hasattr(agent, "memory") and hasattr(agent.memory, "steps"):
        for previous_step in agent.memory.steps:
            if isinstance(previous_step, ActionStep) and previous_step.step_number <= current_step - 2:
                previous_step.observations_images = None

    png_bytes = driver.get_screenshot_as_png()
    image = Image.open(BytesIO(png_bytes))
    step_log.observations_images = [image.copy()]

    url_info = f"Current url: {driver.current_url}"
    step_log.observations = (
        url_info if step_log.observations is None else step_log.observations + "\n" + url_info
    )

agent = CodeAgent(
    tools=[
        DuckDuckGoSearchTool(),
        open_browser,
        search_in_browser,
        open_wikipedia_page,
        open_wikipedia_and_screenshot,
        go_back,
        close_popups,
        search_item_ctrl_f,
        take_browser_screenshot,
    ],
    model=model,
    additional_authorized_imports=["helium"],
    step_callbacks=[save_screenshot],
    max_steps=20,
    verbosity_level=2,
)

response = agent.run(
    """
    use search_in_browser to search for the character's typical appearance and then describe the costume and makeup.
    after searching, use open_wikipedia_and_screenshot for each comparison character so each screenshot is saved
    immediately after opening that specific Wikipedia page.
    Describe the costume and makeup that the comic character in these photos is wearing and return the description.
    In the final answer, also describe what each saved Wikipedia screenshot shows about the character's appearance.
    Tell me if the guest is The Joker or Wonder Woman or spider man search for all the characters.
    do not open all Wikipedia pages first and screenshot later.
    """,
    images=images,
)

print(response)