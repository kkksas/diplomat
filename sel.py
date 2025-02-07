from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup


browser = webdriver.Chrome()
# 'https://xn--90adear.xn--p1ai/check/auto' –  ГИБДД.РФ
browser.get('https://xn--90adear.xn--p1ai/check/auto')