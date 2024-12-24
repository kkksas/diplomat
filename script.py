def splitter(path: str, keyword: str) -> str:
    with open(path, 'r') as file:
        content = file.read()
    return content.split(keyword)
splitter('./desc.txt', 'ТУР')