import os


class LocalStorageService:
    def solution_path(self) -> str:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

    def data_path(self) -> str:
        path = os.path.join(self.solution_path(), "data")

        if not os.path.exists(path):
            os.makedirs(path)

        return path
