from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import joblib
import pandas as pd

from database import Base, engine, SessionLocal
from models import MeterData
from schemas import MeterInput

# Create DB tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Smart Meter AI Backend")

# CORS (important for HTML)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = joblib.load("../model/hes_ocsvm_model.  pkl")
scaler = joblib.load("../model/hes_scaler.pkl")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ======================
# HES INGEST API
# ======================
@app.post("/hes/ingest")
def ingest(data: MeterInput, db: Session = Depends(get_db)):

    df = pd.DataFrame([{
        "daily_kwh": data.daily_kwh,
        "cumulative_kwh": data.cumulative_kwh,
        "plan_kwh": data.plan_kwh,
        "remaining_credit": data.remaining_credit,
        "usage_ratio": data.usage_ratio
    }])

    X_scaled = scaler.transform(df)
    pred = model.predict(X_scaled)[0]
    score = model.decision_function(X_scaled)[0]

    status = "Anomaly" if pred == -1 else "Normal"

    record = MeterData(
        consumer_id=data.consumer_id,
        daily_kwh=data.daily_kwh,
        cumulative_kwh=data.cumulative_kwh,
        plan_kwh=data.plan_kwh,
        remaining_credit=data.remaining_credit,
        usage_ratio=data.usage_ratio,
        anomaly_score=float(score),
        status=status
    )

    db.add(record)
    db.commit()

    return {
        "consumer_id": data.consumer_id,
        "status": status,
        "anomaly_score": round(score, 4)
    }

# ======================
# DASHBOARD API
# ======================
@app.get("/alerts")
def get_alerts(db: Session = Depends(get_db)):
    return db.query(MeterData)\
        .filter(MeterData.status == "Anomaly")\
        .order_by(MeterData.id.desc())\
        .all()