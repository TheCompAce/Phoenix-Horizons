class Tag:
    def __init__(self, name, path, line_number, code_type):
        self.name = name
        self.path = path
        self.line_number = line_number
        self.code_type = code_type

    def to_dict(self):
        return {
            'name': self.name,
            'path': self.path,
            'line_number': self.line_number,
            'code_type': self.code_type,
        }