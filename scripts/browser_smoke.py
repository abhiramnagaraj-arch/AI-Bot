"""Run the real browser/API/parser workflow with a deterministic test model."""

import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
from playwright.sync_api import sync_playwright


def main():
    with tempfile.TemporaryDirectory(prefix="folio-browser-") as directory:
        env = {**os.environ, "BROWSER_TEST_DATA": directory, "OPENBLAS_NUM_THREADS": "1"}
        with open(Path(directory) / "server.log", "w") as log:
            server = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "tests.browser_app:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "18091",
                    "--no-access-log",
                ],
                env=env,
                stdout=log,
                stderr=log,
            )
            try:
                for _ in range(60):
                    try:
                        if httpx.get("http://127.0.0.1:18091/health/live", timeout=1).is_success:
                            break
                    except httpx.HTTPError:
                        pass
                    if server.poll() is not None:
                        raise RuntimeError(Path(directory, "server.log").read_text())
                    time.sleep(1)
                else:
                    raise RuntimeError("Browser test server did not start")
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch(
                        executable_path=os.environ.get("CHROME_PATH", "/usr/bin/google-chrome"),
                        headless=True,
                        args=["--no-sandbox"],
                    )
                    page = browser.new_page(viewport={"width": 1440, "height": 1000})
                    errors = []
                    page.on("pageerror", lambda error: errors.append(str(error)))
                    page.goto("http://127.0.0.1:18091")
                    page.locator("#username").fill("alice")
                    page.locator("#access-key").fill("alice-access-key-0123456789")
                    page.locator("#login-button").click()
                    page.locator("#workspace").wait_for(state="visible")
                    page.locator("#file-input").set_input_files(
                        {
                            "name": "Leave policy.txt",
                            "mimeType": "text/plain",
                            "buffer": b"Employees receive 20 days of annual leave. Requests go to the manager.",
                        }
                    )
                    page.get_by_text(re.compile("passages · Ready")).wait_for(timeout=60000)
                    page.locator("#question").fill("How much annual leave do employees receive?")
                    page.locator("#send-button").click()
                    page.get_by_text(
                        "Employees receive 20 days of annual leave. [S1]", exact=True
                    ).wait_for(timeout=30000)
                    page.locator(".source-details summary").click()
                    page.locator(".source-card").wait_for(state="visible")
                    page.screenshot(path="/tmp/folio-desktop.png", full_page=True)
                    with page.expect_download() as downloaded:
                        page.locator(".source-card a").click()
                    assert downloaded.value.suggested_filename == "Leave policy.txt"
                    page.reload()
                    page.get_by_text(
                        "Employees receive 20 days of annual leave. [S1]", exact=True
                    ).wait_for()
                    page.set_viewport_size({"width": 390, "height": 844})
                    page.screenshot(path="/tmp/folio-mobile.png", full_page=True)
                    assert page.evaluate(
                        "document.documentElement.scrollWidth <= window.innerWidth"
                    )
                    page.on("dialog", lambda dialog: dialog.accept())
                    page.locator(".document button").filter(has_text="Delete").click()
                    page.get_by_text("No documents yet.", exact=False).wait_for()
                    assert not errors, errors
                    browser.close()
                    print(
                        "Browser smoke passed: login, upload, indexing, chat, citations, source download, reload, mobile layout, deletion; no JavaScript errors."
                    )
            finally:
                server.terminate()
                try:
                    server.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()


if __name__ == "__main__":
    main()
