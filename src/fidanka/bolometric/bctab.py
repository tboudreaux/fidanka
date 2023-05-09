from fidanka.bolometric.load import load_bol_table, load_bol_table_metadata

class Bolometric:
    def __init__(self, path, FeH):
        self.path = path
        self.metadata = load_bol_table_metadata(path)
        self.FeH = FeH
        self.data = load_bol_table(path)
