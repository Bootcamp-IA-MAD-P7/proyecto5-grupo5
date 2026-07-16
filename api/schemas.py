from pydantic import BaseModel, Field


class CustomerData(BaseModel):
    Gender: str = Field(..., description="Male or Female")
    Senior_Citizen: int = Field(..., alias="Senior Citizen", ge=0, le=1)
    Partner: str = Field(..., description="Yes or No")
    Dependents: str = Field(..., description="Yes or No")
    Tenure_Months: int = Field(..., alias="Tenure Months", ge=1, le=72)
    Phone_Service: str = Field(..., alias="Phone Service", description="Yes or No")
    Multiple_Lines: str = Field(..., alias="Multiple Lines", description="Yes or No")
    Internet_Service: str = Field(..., alias="Internet Service", description="DSL, Fiber optic, or No")
    Online_Security: str = Field(..., alias="Online Security", description="Yes or No")
    Online_Backup: str = Field(..., alias="Online Backup", description="Yes or No")
    Device_Protection: str = Field(..., alias="Device Protection", description="Yes or No")
    Tech_Support: str = Field(..., alias="Tech Support", description="Yes or No")
    Streaming_TV: str = Field(..., alias="Streaming TV", description="Yes or No")
    Streaming_Movies: str = Field(..., alias="Streaming Movies", description="Yes or No")
    Contract: str = Field(..., description="Month-to-month, One year, or Two year")
    Paperless_Billing: str = Field(..., alias="Paperless Billing", description="Yes or No")
    Payment_Method: str = Field(..., alias="Payment Method", description="Bank transfer (automatic), Credit card (automatic), Electronic check, or Mailed check")
    Monthly_Charges: float = Field(..., alias="Monthly Charges", ge=0)
    Total_Charges: float = Field(..., alias="Total Charges", ge=0)


class HealthResponse(BaseModel):
    status: str = "ok"
    model: str
    model_loaded: bool


class FeedbackRequest(BaseModel):
    prediction_id: int
    correct: bool
    true_label: int | None = None


class IngestRequest(BaseModel):
    data: dict | list
    true_label: int | None = None


class MetricsResponse(BaseModel):
    total_predictions: int
    feedback_received: int
    correct_predictions: int
    accuracy: float | None
    churn_predictions: int
