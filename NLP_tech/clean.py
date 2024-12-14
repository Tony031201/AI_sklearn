def clean(text):
    # drop html label
    text['text'] = re.sub(r"<.*?>",'',text['text'])
    # drop non alphabet
    text['text'] = re.sub(r"[^a-zA-Z]",' ',text['text'])
    text['text'] = text['text'].lower().strip()
    return text