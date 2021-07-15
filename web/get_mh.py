#!/usr/bin/env python
# -*- coding:utf-8 -*-
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

mobile_emulation = {
    "deviceMetrics": {"width": 700, "height": 800, "pixelRatio": 3.0}, #  定义设备高宽，像素比
    "userAgent": "Mozilla/5.0 (Linux; Android 8.0.1; en-us; Nexus 5 Build/JOP40D) " # 通过UA来模拟
                 "AppleWebKit/535.19 (KHTML, like Gecko) Chrome/18.0.1025.166 Mobile Safari/535.19"}

option = webdriver.ChromeOptions()
# option.add_argument("--user-data-dir="+r"C:/Users/jsyzdlf/AppData/Local/Google/Chrome/User Data/")
option.add_extension('E:/tool/crx/ifebaancnnlmdehpiojjndcolgbcjcll_v0.9.1.crx')
option.add_experimental_option("mobileEmulation", mobile_emulation)
driver = webdriver.Chrome(options=option)

# driver.get("http://m.baidu.com")
driver.get("https://m.svipmh.com")
