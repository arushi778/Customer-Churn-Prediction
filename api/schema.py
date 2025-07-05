from pydantic import BaseModel

class ChurnRequest(BaseModel):
    gender: str
    senior_citizen: int
    tenure: int
    monthly_charges: float
    total_charges: float
    contract: str
    payment_method: str
    internet_service: str

class ChurnResponse(BaseModel):
    churn: bool
    probability: float