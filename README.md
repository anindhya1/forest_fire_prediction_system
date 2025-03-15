# Forest Fire Prediction System

A cloud-based application that predicts forest fire risk based on environmental factors such as temperature, humidity, wind speed, and rainfall patterns.

## Features

- Analyzes multiple environmental parameters to assess fire risk
- Provides real-time risk classification (Low, Moderate, High, Extreme)
- Stores historical predictions for trend analysis
- Responsive web interface accessible from any device
- Deployed on AWS Elastic Beanstalk for scalability and reliability

## Technology Stack

- **Backend**: Python Flask
- **Frontend**: HTML, CSS, JavaScript
- **Hosting**: AWS Elastic Beanstalk
- **Data Analysis**: Pandas, NumPy
- **Data Storage**: CSV-based (with optional upgrade path to AWS S3/RDS)

## Live Demo

The application is deployed and accessible at:
[http://forest-fire-dev.elasticbeanstalk.com](http://forest-fire-dev.elasticbeanstalk.com)
(Replace with your actual Elastic Beanstalk URL)

## How It Works

The Forest Fire Prediction System uses a weighted algorithm that considers:

1. **Temperature**: Higher temperatures increase the risk of fire ignition and spread
2. **Humidity**: Lower humidity levels make vegetation more flammable
3. **Wind Speed**: Stronger winds accelerate fire spread and increase intensity
4. **Rainfall**: Recent rainfall reduces fire risk by increasing moisture content
5. **Drought Conditions**: Extended periods without rain significantly increase risk

The algorithm assigns risk scores to each factor and calculates an overall risk level.

## Local Development Setup

### Prerequisites

- Python 3.11 or higher
- pip package manager
- Git

### Installation

1. Clone the repository
   ```bash
   git clone https://github.com/anindhya1/forest_fire_prediction_system.git
   cd forest_fire_prediction_system
   ```

2. Create a virtual environment
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application locally
   ```bash
   python application.py
   ```

5. Open your browser and navigate to `http://localhost:8000`

## AWS Deployment

This application is configured for deployment on AWS Elastic Beanstalk:

1. Install the EB CLI
   ```bash
   pip install awsebcli
   ```

2. Configure AWS credentials
   ```bash
   aws configure
   ```

3. Initialize and deploy
   ```bash
   eb init -p python-3.11 forest-fire-predictor
   eb create forest-fire-dev
   ```

## Future Enhancements

- Integration with real-time weather APIs
- Machine learning-based predictive models
- Satellite imagery analysis for vegetation assessment
- User authentication and personalized monitoring areas
- Alert system for high-risk conditions
- Mobile application for field use

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Weather data sources
- AWS for cloud infrastructure
- Forest fire research organizations for risk assessment methodologies
