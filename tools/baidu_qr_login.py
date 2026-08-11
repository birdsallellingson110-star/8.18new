#!/usr/bin/env python3
import json
import os
import time
from pathlib import Path

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service


out_dir = Path("/mnt/data/cjydata/baidu_qr_login_h36m")
out_dir.mkdir(parents=True, exist_ok=True)
os.chmod(out_dir, 0o700)

options = Options()
options.add_argument("-headless")
options.add_argument("--width=1280")
options.add_argument("--height=960")

service = Service("/home/lixiaob/cjy/tools/geckodriver-0.37.1/geckodriver")
driver = webdriver.Firefox(service=service, options=options)
try:
    driver.get("https://pan.baidu.com/")
    time.sleep(6)

    # The home page may show a login entry before rendering the QR panel.
    for xpath in (
        "//*[contains(normalize-space(.), '登录') and (self::button or self::a or @role='button')]",
        "//*[contains(@class, 'login')]",
    ):
        try:
            candidates = driver.find_elements(By.XPATH, xpath)
            if candidates:
                candidates[0].click()
                time.sleep(4)
                break
        except Exception:
            continue

    driver.save_screenshot(str(out_dir / "login_page.png"))
    (out_dir / "page_source.html").write_text(driver.page_source, encoding="utf-8")

    deadline = time.time() + 600
    while time.time() < deadline:
        cookies = driver.get_cookies()
        names = {item["name"] for item in cookies}
        if "BDUSS" in names:
            cookie_file = out_dir / "cookies.json"
            cookie_file.write_text(json.dumps(cookies), encoding="utf-8")
            os.chmod(cookie_file, 0o600)
            print("LOGIN_SUCCESS", flush=True)
            break
        driver.save_screenshot(str(out_dir / "login_page.png"))
        time.sleep(3)
    else:
        print("LOGIN_TIMEOUT", flush=True)
finally:
    driver.quit()
