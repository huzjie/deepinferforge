# deepinferforge - zero-dependency reference runtime
FROM python:3.12-slim

WORKDIR /app

COPY deepinferforge ./deepinferforge
COPY README.md LICENSE pyproject.toml ./

EXPOSE 8421

ENTRYPOINT ["python", "-m", "deepinferforge"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8421"]
