import os
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="High-Value Customer Pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.join(BASE_DIR, 'ml')

class PipelineState:
    models = {}
    preprocessor = None
    feature_importances = {}
    metrics = {}

state = PipelineState()

@app.on_event("startup")
def load_artifacts():
    try:
        state.models["Random Forest"] = joblib.load(os.path.join(ML_DIR, 'random_forest.joblib'))
        state.models["Logistic Regression"] = joblib.load(os.path.join(ML_DIR, 'logistic_regression.joblib'))
        state.models["SVM"] = joblib.load(os.path.join(ML_DIR, 'svm.joblib'))
        state.preprocessor = joblib.load(os.path.join(ML_DIR, 'preprocessor.joblib'))
        
        with open(os.path.join(ML_DIR, 'metrics.json'), 'r') as f:
            state.metrics = json.load(f)
        with open(os.path.join(ML_DIR, 'feature_importances.json'), 'r') as f:
            state.feature_importances = json.load(f)
    except Exception as e:
        print(f"Warning: Could not load ML artifacts: {e}")

class CustomerData(BaseModel):
    age: int
    account_age_months: int
    total_purchases: int
    avg_order_value: float
    days_since_last_purchase: float
    cart_abandonment_rate: float
    product_reviews_count: int
    avg_review_rating: float
    email_opens: int
    bounce_rate: float
    customer_segment: str
    device_type: str
    country: str
    has_promo_code: int
    model_name: str = "Random Forest"

@app.post("/predict")
def predict(data: CustomerData):
    if data.model_name not in state.models:
        raise HTTPException(status_code=400, detail="Model not found")
        
    model = state.models[data.model_name]
    
    input_df = pd.DataFrame([{
        "age": data.age,
        "account_age_months": data.account_age_months,
        "total_purchases": data.total_purchases,
        "avg_order_value": data.avg_order_value,
        "days_since_last_purchase": data.days_since_last_purchase,
        "cart_abandonment_rate": data.cart_abandonment_rate,
        "product_reviews_count": data.product_reviews_count,
        "avg_review_rating": data.avg_review_rating,
        "email_opens": data.email_opens,
        "bounce_rate": data.bounce_rate,
        "customer_segment": data.customer_segment,
        "device_type": data.device_type,
        "country": data.country,
        "has_promo_code": data.has_promo_code
    }])
    
    try:
        proc_data = state.preprocessor.transform(input_df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {str(e)}")

    prediction = int(model.predict(proc_data)[0])
    probability = float(model.predict_proba(proc_data)[0][1])

    # Explainable AI logic
    top_3 = list(state.feature_importances.items())[:3]
    top_names = [k.replace('num__', '').replace('cat__', '').replace('_', ' ').title() for k, v in top_3]
    
    reason = "Unknown"
    is_high = prediction == 1
    
    if is_high:
        if data.total_purchases > 10:
            segment = "High Spenders (VIP)"
            reason = f"High engagement across {top_names[0]} and {top_names[1]} solidifies this customer's VIP status, drastically increasing high-value purchase likelihood."
        else:
            segment = "Rising Stars"
            reason = f"Recent activities like {top_names[0]} strongly predict future high-value behavior despite moderate purchasing history."
            
        recs = [
            "🏆 Suggest Premium Elite offers & loyalty accelerators",
            "💎 Assign a dedicated concierge or VIP support line",
            "📈 Target strictly with upselling/cross-selling strategies"
        ]
    else:
        if data.cart_abandonment_rate > 0.6 or data.bounce_rate > 0.5:
            segment = "At-Risk Customers"
            reason = f"High friction noted in {top_names[0]} and bounce metrics indicates churn risk and limits value."
            recs = [
                "🚀 Send immediate cart-recovery workflows with 15% discount",
                "🎯 Deploy re-engagement surveys to determine friction points",
                "📉 Suppress from high-ticket ad campaigns temporarily"
            ]
        else:
            segment = "Occasional Buyers"
            reason = f"Customer shows steady but low-velocity patterns constrained by {top_names[0]}, requiring nurturing."
            recs = [
                "🎁 Offer entry-level bundles to increase average order value",
                "📧 Nurture with educational/value-add email sequences",
                "🏷️ Provide a time-limited discount to accelerate the next purchase"
            ]
            
    # Bonus Feature: Customer Lifetime Value (CLV)
    purchases_per_year = (data.total_purchases / max(1, data.account_age_months)) * 12
    lifespan_estimate = 3 if is_high else 1.5
    clv = round((data.avg_order_value * purchases_per_year) * lifespan_estimate, 2)

    return {
        "prediction": prediction,
        "probability": probability,
        "segment": segment,
        "explainability": reason,
        "recommendations": recs,
        "top_influencing_features": top_names,
        "estimated_clv_usd": clv
    }
