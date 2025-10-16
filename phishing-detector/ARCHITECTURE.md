# Phishing Detector - Project Overview

## Project Structure

```
phishing-detector/
├── api/                          # FastAPI REST API
│   ├── main.py                  # API entry point
│   ├── schemas.py               # Pydantic models
│   └── routers/
│       └── detection.py         # Detection endpoints
├── web/
│   └── app.py                   # Streamlit web interface
├── src/                         # Core source code
│   ├── preprocessing/           # Data preprocessing
│   │   ├── email_processor.py  # Email text processing
│   │   └── url_extractor.py    # URL extraction
│   ├── models/                  # ML models
│   │   ├── email_classifier.py # DistilBERT classifier
│   │   ├── url_classifier.py   # URL classifier
│   │   └── detector.py         # Integrated detector
│   ├── features/                # Feature extraction
│   │   └── url_features.py     # URL feature engineering
│   └── utils/                   # Utilities
│       ├── config.py           # Configuration management
│       └── helpers.py          # Helper functions
├── tests/                       # Test suite
│   ├── test_preprocessing.py   # Preprocessing tests
│   ├── test_models.py          # Model tests
│   └── test_api.py             # API integration tests
├── scripts/                     # Utility scripts
│   ├── prepare_data.py         # Data preparation
│   ├── train_model.py          # Model training
│   └── evaluate_model.py       # Model evaluation
├── config/                      # Configuration files
│   ├── config.yaml             # App configuration
│   └── model_config.json       # Model hyperparameters
├── data/                        # Data storage
│   ├── raw/                    # Raw datasets
│   ├── processed/              # Processed data
│   └── external/               # External data sources
├── models/                      # Trained models
│   ├── email_classifier/       # Email models
│   ├── url_classifier/         # URL models
│   └── checkpoints/            # Training checkpoints
├── notebooks/                   # Jupyter notebooks (optional)
├── logs/                        # Application logs
├── results/                     # Evaluation results
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
├── LICENSE                      # MIT License
├── .gitignore                  # Git ignore rules
├── Dockerfile                   # Docker configuration
├── docker-compose.yml          # Docker Compose config
└── setup.ps1                   # Quick setup script
```

## Key Components

### 1. Email Classification (DistilBERT)
- Transformer-based deep learning model
- Fine-tuned on phishing email datasets
- Analyzes text patterns and linguistic features
- Returns phishing probability score

### 2. URL Analysis (Random Forest)
- Feature-based classification
- Analyzes URL structure and patterns
- Detects suspicious TLDs, IP addresses, entropy
- Fast inference (<10ms per URL)

### 3. Integrated Detection
- Combines email and URL analysis
- Weighted scoring (60% email, 40% URL)
- Risk categorization (HIGH/MEDIUM/LOW/SAFE)
- Detailed explanations and indicators

### 4. REST API (FastAPI)
- `/api/v1/detect` - Single email detection
- `/api/v1/detect/batch` - Batch processing
- `/api/v1/health` - Health check
- `/api/v1/stats` - Model statistics
- Auto-generated documentation at `/docs`

### 5. Web Interface (Streamlit)
- User-friendly interface
- Real-time analysis
- Visual risk scoring
- Example emails
- Detailed explanations

## Development Workflow

### Setup
```powershell
# Run quick setup
.\setup.ps1

# Or manual setup
python -m venv venv
.\venv\Scripts\Activate
pip install -r requirements.txt
```

### Running
```powershell
# Start API
cd api
python main.py

# Start Web (new terminal)
streamlit run web\app.py

# Run Tests
pytest tests\ -v --cov=src
```

### Training Models
```powershell
# Prepare data
python scripts\prepare_data.py

# Train email classifier
python scripts\train_model.py

# Evaluate
python scripts\evaluate_model.py
```

## Testing

### Unit Tests
- `test_preprocessing.py` - Data preprocessing
- `test_models.py` - Model functionality

### Integration Tests
- `test_api.py` - API endpoints
- End-to-end detection pipeline

### Run Tests
```powershell
# All tests
pytest tests\ -v

# With coverage
pytest tests\ --cov=src --cov-report=html

# Specific test file
pytest tests\test_models.py -v
```

## Configuration

### config/config.yaml
- API settings (host, port)
- Model paths
- Detection thresholds
- Logging configuration

### config/model_config.json
- Model hyperparameters
- Training configuration
- Feature specifications

## Deployment

### Local
```powershell
# API
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Web
streamlit run web\app.py
```

### Docker
```powershell
# Build and run
docker-compose up -d

# Access
# API: http://localhost:8000
# Web: http://localhost:8501
```

### Cloud
- AWS: EC2, Lambda, SageMaker
- Google Cloud: Cloud Run, Vertex AI
- Azure: App Service
- Hugging Face Spaces

## Performance

### Metrics
- Accuracy: >92%
- Precision: >90%
- Recall: >88%
- F1-Score: >90%
- Inference: <500ms per email

### Optimization
- Model quantization for faster inference
- Batch processing for throughput
- Caching for frequently analyzed URLs
- Async API for concurrency

## Security

### API Security
- Input validation
- Rate limiting (recommended)
- CORS configuration
- Error handling

### Data Privacy
- No data storage by default
- Configurable logging
- Secure model storage

## Maintenance

### Monitoring
- API health checks
- Model performance tracking
- Error logging
- Usage statistics

### Updates
- Regular model retraining
- Dataset updates
- Dependency updates
- Security patches

## Contributing

1. Fork repository
2. Create feature branch
3. Add tests
4. Update documentation
5. Submit pull request

## License

MIT License - see LICENSE file

## Support

- GitHub Issues
- Documentation
- API Reference: /docs
