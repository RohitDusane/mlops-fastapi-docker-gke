
install:
	pip install -r requirements.txt

dev:
	pip install -r requirements-dev.txt

train:
	python main.py

predict:
	python app/app.py

test:
	pytest tests/

lint:
	ruff check src/

format:
	black src/

docker-build:
	docker build -t diabetes-risk .

docker-run:
	docker run -p 8000:8000 diabetes-risk

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -name "*.pyc" -delete