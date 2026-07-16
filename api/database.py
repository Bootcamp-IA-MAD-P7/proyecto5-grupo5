import os
from pathlib import Path
from datetime import datetime, timezone

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Boolean, Text
from sqlalchemy.orm import declarative_base, Session

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{Path(__file__).resolve().parent.parent / 'data' / 'churn.db'}")

engine = create_engine(DATABASE_URL)
Base = declarative_base()


class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    input_data = Column(Text)
    churn_prediction = Column(Integer)
    churn_probability = Column(Float)
    model_name = Column(String)
    feedback = Column(Boolean, nullable=True)


class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True)
    prediction_id = Column(Integer)
    correct = Column(Boolean)
    true_label = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class IngestedData(Base):
    __tablename__ = "ingested_data"
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    data = Column(Text)
    true_label = Column(Integer, nullable=True)


Base.metadata.create_all(engine)


def get_session():
    return Session(engine)


def save_prediction(input_data: str, prediction: int, probability: float, model_name: str) -> int:
    with get_session() as session:
        p = Prediction(
            input_data=input_data,
            churn_prediction=prediction,
            churn_probability=probability,
            model_name=model_name,
        )
        session.add(p)
        session.commit()
        return p.id


def save_feedback(prediction_id: int, correct: bool, true_label: int | None = None):
    with get_session() as session:
        f = Feedback(
            prediction_id=prediction_id,
            correct=correct,
            true_label=true_label,
        )
        session.add(f)
        pred = session.query(Prediction).filter_by(id=prediction_id).first()
        if pred:
            pred.feedback = correct
        session.commit()


def save_ingested_data(data: str, true_label: int | None = None):
    with get_session() as session:
        d = IngestedData(data=data, true_label=true_label)
        session.add(d)
        session.commit()


def get_metrics():
    with get_session() as session:
        total = session.query(Prediction).count()
        feedback_count = session.query(Prediction).filter(Prediction.feedback.isnot(None)).count()
        correct = session.query(Prediction).filter(Prediction.feedback == True).count()
        churn_count = session.query(Prediction).filter(Prediction.churn_prediction == 1).count()
        return {
            "total_predictions": total,
            "feedback_received": feedback_count,
            "correct_predictions": correct,
            "accuracy": round(correct / feedback_count, 4) if feedback_count else None,
            "churn_predictions": churn_count,
        }
