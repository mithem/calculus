from functions import Function
import math

class sin(Function):

    def evaluate(self, x: float) -> float:
        return math.sin(x)

class cos(Function):

    def evaluate(self, x: float) -> float:
        return math.cos(x)

class tan(Function):
    
    def evaluate(self, x: float) -> float:
        return math.tan(x)
