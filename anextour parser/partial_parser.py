from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
import numpy as np
import pandas as pd
import time
from selenium.common.exceptions import StaleElementReferenceException, ElementClickInterceptedException, NoSuchElementException
import re

service = Service(executable_path=ChromeDriverManager().install())
options = webdriver.ChromeOptions()
options.add_argument(r"user-data-dir=C:\\Users\\kosty\AppData\\Local\\Google\\Chrome\\User Data\\Profile 1")
options.add_argument('--ignore-certificate-errors')
options.add_argument('--ignore-ssl-errors')
driver = webdriver.Chrome(service=service, options=options)

def parse_tour_page(driver, url = 'https://anextour.ru/excursion-tours/russia/zolotaya-moskva-leto'):
    #'https://anextour.ru/excursion-tours/russia/zolotaya-moskva-leto'
    driver.get(url)
    name = driver.find_element("xpath", '//main/section[1]/div/div[1]/div[1]/div/h1').text.splitlines()[1]
    desk_time = driver.find_element("xpath", "//main//div[@class='w-full inherit-all lg:pr-68 lg:text-16 md:text-14 sm:text-14']").text.strip()
    desk_time = re.split(r"(?:\r?\n){1,}", desk_time)
    if len(desk_time)>2:
        haha = 1
    includes = []
    for i in driver.find_elements("xpath", "//main//div[@class='pb-16']//ul/li/div//h3"):
        includes.append(i.text)
    path = []
    try:
        driver.find_element("xpath", "//main//section[contains(.,'Маршрут тура')]/div[2]//button").click()
    except NoSuchElementException:
        print('all info displayed')   
    path.append(driver.find_element("xpath", "//main//div[@class='wysiwyg-pointsList']").text)
    
    
    try:
        driver.find_element("xpath", "//main//section[contains(.,'Маршрут тура')]//div[@id = ':rqt:']/div[2]/button").click()
        btn = driver.find_elements("xpath", "//main//section[contains(.,'Маршрут тура')]//div[@id = ':rqt:']/div[3]/div/div[2]//button")
        for i in btn[1:]:
            i.click()
            path.append(driver.find_element("xpath", "//main//div[@class='wysiwyg-pointsList']").text)
    except NoSuchElementException:
        print("there is one date variant")
    
    try:
        path_variant_but = driver.find_elements("xpath", "//main/section[2]/div[1]//li/label[@gtm-label='false']")   
        for i in path_variant_but:
            i.click()
            path.append(driver.find_element("xpath", "//main//div[@class='wysiwyg-pointsList']").text)
    except NoSuchElementException:
        print('there only one way')
    output = []
    for i in path:
        i.replace('\n',' ')
        output.append([name, desk_time[0], desk_time[1], includes, i])

    #driver.execute_script("window.history.go(-1)")
    return output

urls = ['https://anextour.ru/excursion-tours/russia/krugosvetka-po-altayu',
        'https://anextour.ru/excursion-tours/russia/altaiskaya-kollekciya',
        'https://anextour.ru/excursion-tours/russia/dagestanskii-ekspress',
        'https://anextour.ru/excursion-tours/russia/vesennie-kanikuly-v-dagestane',
        'https://anextour.ru/excursion-tours/russia/sokrovisha-dagestana',
        'https://anextour.ru/excursion-tours/russia/kaspiiskie-kanikuly',
        'https://anextour.ru/excursion-tours/russia/baikal-khit',
        'https://anextour.ru/excursion-tours/russia/aviatur-po-tryom-ostrovam',
        'https://anextour.ru/excursion-tours/russia/tur-3-ostrova-s-paromom'
        ]
data = []
for url in urls:
    data.extend(parse_tour_page(driver, url))
df = pd.DataFrame(data)
print(df)
df.to_excel('new_findings.xlsx')