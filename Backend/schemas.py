from pydantic import BaseModel

class MeterInput(BaseModel):
    consumer_id: str

    daily_kwh: float
    cumulative_kwh: float
    plan_kwh: float
    remaining_credit: float
    usage_ratio: float