from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api import router

app = FastAPI(title="TrustedAI Evaluation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # 若上生产可限定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
