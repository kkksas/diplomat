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



driver.get('https://anextour.ru/excursion-tours/russia')
bottom_botton = driver.find_element("xpath", "//button[contains(.,'Смотреть ещё')]")
while True:
    try:
        bottom_botton.click()
        time.sleep(0.5)
    except StaleElementReferenceException:
        break
    except ElementClickInterceptedException:
        time.sleep(2)
        bottom_botton.click()
tour_cards = driver.find_elements("xpath", "//li/a[@gtm-label= 'card']")
links = [elem.get_attribute('href') for elem in tour_cards] 
data = []
for i in links:    
    data.extend(parse_tour_page(driver, i))


df = pd.DataFrame(data)
print(df)
df.to_excel('parsed.xlsx')

driver.quit()




"""
проход по регионам
driver.find_element("xpath", "//div[@id = ':R3d8r6m:']/div[2]").click()   
regions =[]
time.sleep(5)
a = driver.find_element("xpath", "//div[@class = 'scroll-vertical-desktop']")
for i in range(30):     
    driver.execute_script(f"arguments[0].scrollBy(0,{39})", a)
    time.sleep(1)
    #regions.extend(driver.find_elements("xpath", "//div[@class = 'scroll-vertical-desktop']/div/div"))
regions = set(regions)
for i in regions:
    print(i.get_attribute("style"))
print('Всего регионов', len(regions))


"""


"""
тащим это
1. Название тура
2) Краткое описание
3) даты
4) Включенные "услуги"
5) Полное описание тура

Хпути для:
1)//main//h1[@data-ptype="468"]/text()[1] - название тура
2)//main//div[@class='w-full inherit-all lg:pr-68 lg:text-16 md:text-14 sm:text-14']/text()[1] - краткое описание тура
3)//main//div[@class='w-full inherit-all lg:pr-68 lg:text-16 md:text-14 sm:text-14']/text()[2] - длительность время тура
4)//main//div[@class='pb-16']//ul/li/div//h3 - в стримость тура включено
5)//main//div[@class='wysiwyg-pointsList'] - маршрут тура проверить на несколько вариантов
6)//main//section[contains(.,'Примечание:')]//ul/li - примечание для прикола

data.append(driver.find_element("class name", "wysiwyg-data").text)
tabs = driver.find_elements("xpath", '//main//ul[@role="tablist"]/li[@aria-selected="false"]')
for elem in tabs:
    elem.click()
    data.append(driver.find_element("class name", "wysiwyg-data").text)




print('lenght',len( data))
for i in data:
    print(i)
    print("**************")
"""



