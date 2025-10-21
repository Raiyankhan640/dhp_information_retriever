# Use official Python image
FROM python:3.9-slim

# Set working directory
# Set working directory
WORKDIR /app

# Copy only requirements first to leverage Docker cache for dependencies
COPY requirements.txt ./

# Install dependencies
RUN pip install --upgrade pip \
	&& pip install --no-cache-dir -r requirements.txt \
	&& rm -rf /root/.cache/pip

# Copy the rest of the application
COPY . /app

# Expose Streamlit default port
EXPOSE 8501

# Set environment variable to avoid Streamlit asking for a browser
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]