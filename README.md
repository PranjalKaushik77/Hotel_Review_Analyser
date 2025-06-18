🏨 Hotel Review Sentiment Analyzer
A comprehensive web application that analyzes hotel customer reviews to extract sentiment insights and identify service-related strengths and weaknesses.

Features
Sentiment Analysis: Classify reviews as positive, negative, or neutral
Service Aspect Extraction: Identify mentions of key service areas (room service, food quality, staff service, etc.)
Interactive Visualizations: Charts and graphs showing sentiment distribution and service insights
Business Insights: Actionable recommendations based on analysis
Multiple Input Methods: Upload CSV files, use sample data, or enter reviews manually
Export Results: Download analysis results as CSV
Service Categories Analyzed
Room Service: Housekeeping, cleaning, towels, bedding, maintenance
Food Quality: Restaurant, breakfast, meals, cuisine, menu
Staff Service: Reception, front desk, concierge, helpfulness
Location: Nearby attractions, transportation, views
Amenities: Pool, gym, spa, wifi, parking, facilities
Value for Money: Pricing, cost-effectiveness
Cleanliness: Hygiene, sanitization standards
Comfort: Bed quality, noise levels, sleep comfort
Installation
Clone or download the project files
Install Python dependencies:
bash
pip install -r requirements.txt
Run the application:
bash
streamlit run app.py
Open your browser to http://localhost:8501
Usage
Option 1: Upload CSV File
Prepare a CSV file with hotel reviews
Upload the file using the file uploader
Select the column containing reviews
Click "Analyze Reviews"
Option 2: Use Sample Data
Select "Use Sample Data" from the sidebar
Click "Analyze Sample Reviews" to see the demo
Option 3: Manual Entry
Select "Enter Manual Reviews"
Type or paste reviews (one per line)
Click "Analyze Manual Reviews"
Understanding the Results
Sentiment Scores
Range: -1 (very negative) to +1 (very positive)
Positive: Score > 0.1
Neutral: Score between -0.1 and 0.1
Negative: Score < -0.1
Service Aspect Scores
Numbers represent how many times each service aspect is mentioned
Higher scores in positive reviews = strength
Higher scores in negative reviews = area for improvement
CSV File Format
Your CSV file should have at least one column containing review text. Example:

csv
review_id,customer_name,review_text,rating
1,John Doe,"Great hotel with excellent service",5
2,Jane Smith,"Room was dirty and staff unhelpful",2
Technical Details
Sentiment Analysis: Uses TextBlob for polarity scoring
Text Processing: NLTK for tokenization, stopword removal, and lemmatization
Visualizations: Plotly for interactive charts
Web Framework: Streamlit for the user interface
Project Structure
hotel-review-analyzer/
├── app.py # Main application file
├── requirements.txt # Python dependencies
├── README.md # This file
└── sample_data.csv # Sample data for testing (optional)
Sample Reviews Format
If creating your own CSV file, ensure reviews are in plain text format:

"The hotel was absolutely wonderful! Great location and excellent service."
"Terrible experience. The room was dirty and service was poor."
"Good hotel overall. Clean rooms and decent location."
Business Applications
Hotel Management
Monitor customer satisfaction trends
Identify service areas needing improvement
Track the impact of service changes over time
Generate reports for stakeholders
Marketing Teams
Understand what customers value most
Identify unique selling points
Address negative feedback in marketing campaigns
Benchmark against competitor reviews
Operations Teams
Focus training on problem areas
Prioritize facility improvements
Optimize resource allocation
Improve service standards
Troubleshooting
Common Issues
NLTK Data Error: The app will automatically download required NLTK data on first run
File Upload Issues: Ensure your CSV file is properly formatted with headers
Memory Issues: For large datasets (>10,000 reviews), consider processing in batches
Visualization Not Loading: Check that all required packages are installed
Performance Tips
For best performance, limit analysis to 5,000 reviews at a time
Ensure review text is clean (remove excessive formatting)
Use UTF-8 encoding for CSV files with special characters
Future Enhancements
Machine learning model training for custom sentiment classification
Multi-language support
Real-time analysis integration with booking platforms
Advanced topic modeling with LDA
Competitor comparison features
Time series analysis for trend tracking
Contributing
This project is designed to be easily extensible. Key areas for enhancement:

Additional Service Categories: Modify the service_keywords dictionary
Advanced NLP: Integrate transformer models for better accuracy
Database Integration: Add persistent storage for historical analysis
API Development: Create REST API endpoints for integration
License
This project is open source and available under the MIT License.

Support
For issues or questions about the Hotel Review Sentiment Analyzer, please check the troubleshooting section above or review the code comments for implementation details.
