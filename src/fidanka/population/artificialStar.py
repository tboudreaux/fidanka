from typing import Union


class artificialStar:
    def __init__(self, calibrated_file: Union[str, None] = None):
        if calibrated_file:
            self.generated = True
        else:
            self.generated = False
        ...

    def completness(self, mag: float, filter: str) -> float:
        ...

    def err(self, mag: float, filter: str) -> float:
        ...

    @property
    def filters(self):
        ...

    def from_calibrated_file(self):
        ...

    def __repr__(self):
        out = f"<artificialStar - generated : {self.generated}>"
        return out
