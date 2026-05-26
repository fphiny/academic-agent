from pydantic import BaseModel


class LoginRequest(BaseModel):
    student_id: str
    password: str


class CourseRecommendationRequest(BaseModel):
    question: str