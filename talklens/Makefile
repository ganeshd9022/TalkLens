.PHONY: install run collect train clean

# Create venv and install dependencies
install:
	python3 -m venv venv
	. venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

# Launch the Streamlit app
run:
	. venv/bin/activate && streamlit run app.py --server.port=8501

# Collect training data for specified signs (e.g. make collect SIGNS="A B C")
SIGNS ?= A B C D E F G H I J
collect:
	. venv/bin/activate && python modules/sign_language/collect_data.py \
	  --signs $(SIGNS) --samples 200 --output data/raw

# Train the sign language classifier
train:
	. venv/bin/activate && python modules/sign_language/train.py \
	  --data data/raw --epochs 80 --batch 32 --lr 1e-3

# Remove caches
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	find . -name "*.pyc" -delete; \
	rm -f talklens.log
