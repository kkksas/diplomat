from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service

service = Service(executable_path=ChromeDriverManager().install())
options = webdriver.ChromeOptions()
options.add_argument(r"user-data-dir=C:\\Users\\kosty\AppData\\Local\\Google\\Chrome\\User Data\\Profile 1")
options.add_argument('--ignore-certificate-errors')
options.add_argument('--ignore-ssl-errors')
driver = webdriver.Chrome(service=service, options=options)

data = []
driver.get('https://anextour.ru/tours/uae/25hours-dubai-one-central')
data.append(driver.find_element("class name", "wysiwyg-data").text)
tabs = driver.find_elements("xpath", '//main//ul[@role="tablist"]/li[@aria-selected="false"]')
for elem in tabs:
    elem.click()
    data.append(driver.find_element("class name", "wysiwyg-data").text)




print('lenght',len( data))
for i in data:
    print(i)
    print("**************")


driver.quit()
