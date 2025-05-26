import os
import json


FAQ_FILE = "faq_data.json"
def load_faq_data():
    if not os.path.exists(FAQ_FILE):
        return {}
    with open(FAQ_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

faq_data = load_faq_data()
items = list(faq_data.items())