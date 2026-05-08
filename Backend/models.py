from sqlalchemy import Column, Integer, String, Float
from database import Base

class MeterData(Base):
    __tablename__ = "meter_data"

    id = Column(Integer, primary_key=True, index=True)
    consumer_id = Column(String, index=True)

    daily_kwh = Column(Float)
    cumulative_kwh = Column(Float)
    plan_kwh = Column(Float)
    remaining_credit = Column(Float)
    usage_ratio = Column(Float)

    anomaly_score = Column(Float)
    status = Column(String)