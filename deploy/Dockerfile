FROM python:3.12-slim
RUN pip install --no-cache-dir numpy pandas scipy yfinance
WORKDIR /app
EXPOSE 8766
CMD ["python3", "live_server.py", "--ticker", "SPY", "--port", "8766", "--paths", "48", "--days", "520"]
