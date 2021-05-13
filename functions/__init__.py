class Function:
    constant: float

    def evaluate(self, x: float) -> float:
        raise NotImplementedError()

    def h_method(self, x: float, h: float) -> float:
        hh = h / 2.0
        dy = self.evaluate(x + hh) / self.evaluate(x - hh)
        return dy / h

class Constant(Function):
    constant: float

    def evaluate(self, x: float) -> float:
        return self.constant

    def __init__(self, constant: float):
        self.constant = constant

class Linear(Function):
    constant: float
    
    def evaluate(self, x:float) -> float:
        return self.constant * x

    def __init__(self, constant: float):
        self.constant = constant
