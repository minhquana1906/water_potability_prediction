from pydantic import BaseModel


class Water(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float


# you can define your own default values for the input parameters
class WaterTest(BaseModel):
    ph: float = 7.0
    Hardness: float = 150.0
    Solids: float = 20000.0
    Chloramines: float = 7.5
    Sulfate: float = 350.0
    Conductivity: float = 400.0
    Organic_carbon: float = 10.0
    Trihalomethanes: float = 80.0
    Turbidity: float = 4.0
