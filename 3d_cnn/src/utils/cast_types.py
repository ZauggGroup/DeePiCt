def string_to_list(string: str, separator: str) -> list:
    return [int(value) for value in string.split(separator)]
