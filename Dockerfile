# NeuroPredict — self-contained container.
#
# Build:   docker build -t neuropredict .
# Run:     docker run -p 8000:8000 neuropredict
# Open:    http://localhost:8000
#
# The demo model is trained during the build, so the container starts instantly
# and needs no account, token, or network access at runtime.

FROM python:3.12-slim

WORKDIR /app

# Install Python dependencies first (better layer caching).
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the project.
COPY . .

# Train the small demo model at build time so the app works out of the box.
RUN python scripts/train_demo.py

EXPOSE 8000

CMD ["uvicorn", "webapp.main:app", "--host", "0.0.0.0", "--port", "8000"]
