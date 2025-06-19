from bs4 import BeautifulSoup


def load_html(file):
    content = file.read().decode("utf-8")
    soup = BeautifulSoup(content, "html.parser")
    return soup.get_text()
