import os
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Directory for saving outputs
OUTPUT_DIR = 'outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

class MarketEntryAnalyzer:
    def __init__(self):
        self.setup_logging()
        self.prepare_market_data()
        self.prepare_historical_data()
        self.prepare_risk_data()

    def setup_logging(self):
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)

    def prepare_historical_data(self):
        """Prepare historical data for forecasting."""
        self.historical_data = {
            'Japan': {
                'GDP': [5.16, 4.85, 4.39, 4.92, 4.87, 4.95, 5.08, 4.97, 4.94, 4.23],
                'Population': [127.3, 127.1, 126.9, 126.7, 126.5, 126.3, 126.1, 125.8, 125.5, 125.2],
                'Consumer_Spending': [2.98, 2.89, 2.82, 2.85, 2.88, 2.91, 2.94, 2.75, 2.82, 2.88],
                'Economic_Growth': [1.6, 0.4, 1.2, 0.5, 2.2, 0.3, 0.7, -4.5, 1.6, 1.0]
            },
            'Brazil': {
                'GDP': [2.47, 2.46, 1.80, 1.80, 2.06, 1.92, 1.88, 1.45, 1.61, 1.84],
                'Population': [201.0, 202.8, 204.5, 206.2, 207.8, 209.3, 210.8, 212.2, 213.6, 214.3],
                'Consumer_Spending': [1.52, 1.48, 1.12, 1.15, 1.28, 1.24, 1.22, 0.98, 1.08, 1.15],
                'Economic_Growth': [3.0, 0.5, -3.5, -3.3, 1.3, 1.8, 1.2, -3.9, 4.6, 2.9]
            },
            'France': {
                'GDP': [2.81, 2.85, 2.43, 2.47, 2.59, 2.78, 2.73, 2.63, 2.96, 2.78],
                'Population': [65.8, 66.2, 66.5, 66.7, 66.9, 67.2, 67.3, 67.4, 67.6, 67.8],
                'Consumer_Spending': [1.62, 1.64, 1.42, 1.44, 1.52, 1.64, 1.61, 1.48, 1.72, 1.65],
                'Economic_Growth': [0.6, 1.0, 1.1, 1.1, 2.3, 1.9, 1.8, -7.9, 6.8, 2.5]
            },
            'Canada': {
                'GDP': [1.84, 1.80, 1.56, 1.53, 1.65, 1.72, 1.74, 1.64, 1.99, 2.00],
                'Population': [35.2, 35.5, 35.8, 36.2, 36.5, 37.0, 37.6, 38.0, 38.2, 38.5],
                'Consumer_Spending': [1.05, 1.02, 0.92, 0.91, 0.98, 1.02, 1.04, 0.96, 1.15, 1.18],
                'Economic_Growth': [2.3, 2.9, 0.7, 1.1, 3.0, 2.8, 1.9, -5.2, 4.5, 3.4]
            }
        }

    def prepare_market_data(self):
        """Prepare current market data."""
        self.current_data = {
            'Japan': {
                'GDP': 4.23,
                'Population': 125.2,
                'Consumer_Spending': 2.88,
                'Economic_Growth': 1.0,
                'Market_Score': 88,
                'Strengths': [
                    'Advanced Infrastructure and Technology',
                    'Strong Financial System',
                    'High Innovation Capacity'
                ],
                'Weaknesses': [
                    'Slow Economic Growth',
                    'Complex Business Regulations',
                    'Limited Market Expansion Potential'
                ]
            },
            'Brazil': {
                'GDP': 1.84,
                'Population': 214.3,
                'Consumer_Spending': 1.15,
                'Economic_Growth': 2.9,
                'Market_Score': 82,
                'Strengths': [
                    'Large Consumer Market',
                    'Growing Middle Class',
                    'Rich Natural Resources'
                ],
                'Weaknesses': [
                    'Complex Regulatory Environment',
                    'Infrastructure Gaps',
                    'Bureaucratic Challenges'
                ]
            },
            'France': {
                'GDP': 2.78,
                'Population': 67.8,
                'Consumer_Spending': 1.65,
                'Economic_Growth': 2.5,
                'Market_Score': 84,
                'Strengths': [
                    'Strong Infrastructure Network',
                    'Skilled Workforce',
                    'Strategic EU Market Position'
                ],
                'Weaknesses': [
                    'High Labor Costs',
                    'Complex Labor Laws',
                    'Moderate Economic Growth'
                ]
            },
            'Canada': {
                'GDP': 2.00,
                'Population': 38.5,
                'Consumer_Spending': 1.18,
                'Economic_Growth': 3.4,
                'Market_Score': 83,
                'Strengths': [
                    'Strong Legal Framework',
                    'Stable Financial System',
                    'High-Quality Infrastructure'
                ],
                'Weaknesses': [
                    'Small Domestic Market',
                    'High Business Operating Costs',
                    'Weather-Related Challenges'
                ]
            }
        }

    def prepare_risk_data(self):
        """Prepare comprehensive risk assessment data."""
        self.risk_factors = {
            'Japan': {
                'Political_Risk': {
                    'score': 15,
                    'factors': [
                        'Stable political environment',
                        'Strong regulatory framework',
                        'Occasional policy shifts affecting business'
                    ]
                },
                'Economic_Risk': {
                    'score': 25,
                    'factors': [
                        'Deflation concerns',
                        'High public debt',
                        'Aging population impact'
                    ]
                },
                'Operational_Risk': {
                    'score': 20,
                    'factors': [
                        'Natural disaster vulnerability',
                        'High operating costs',
                        'Complex business culture'
                    ]
                },
                'Technical_Risk': {
                    'score': 15,
                    'factors': [
                        'Advanced digital infrastructure',
                        'High cybersecurity standards',
                        'Strong technological innovation',
                        'Skilled IT workforce'
                    ]
                }
            },
            'Brazil': {
                'Political_Risk': {
                    'score': 35,
                    'factors': [
                        'Political instability',
                        'Frequent regulatory changes',
                        'Corruption concerns'
                    ]
                },
                'Economic_Risk': {
                    'score': 30,
                    'factors': [
                        'Currency volatility',
                        'Inflation risk',
                        'Economic policy uncertainty'
                    ]
                },
                'Operational_Risk': {
                    'score': 40,
                    'factors': [
                        'Infrastructure limitations',
                        'Complex tax system',
                        'Security concerns'
                    ]
                },
                'Technical_Risk': {
                    'score': 35,
                    'factors': [
                        'Digital infrastructure gaps',
                        'Cybersecurity vulnerabilities',
                        'Limited tech talent pool',
                        'Uneven technological adoption'
                    ]
                }
            },
            'France': {
                'Political_Risk': {
                    'score': 20,
                    'factors': [
                        'Labor union influence',
                        'EU regulatory compliance',
                        'Social movement impact'
                    ]
                },
                'Economic_Risk': {
                    'score': 25,
                    'factors': [
                        'High tax burden',
                        'EU economic dependencies',
                        'Labor market rigidity'
                    ]
                },
                'Operational_Risk': {
                    'score': 15,
                    'factors': [
                        'Strike frequency',
                        'Administrative complexity',
                        'High labor costs'
                    ]
                },
                'Technical_Risk': {
                    'score': 20,
                    'factors': [
                        'Good digital infrastructure',
                        'Moderate cybersecurity framework',
                        'Growing tech innovation',
                        'Competitive IT sector'
                    ]
                }
            },
            'Canada': {
                'Political_Risk': {
                    'score': 10,
                    'factors': [
                        'Stable political system',
                        'Strong property rights',
                        'Transparent regulations'
                    ]
                },
                'Economic_Risk': {
                    'score': 20,
                    'factors': [
                        'US economic dependency',
                        'Housing market volatility',
                        'Resource price sensitivity'
                    ]
                },
                'Operational_Risk': {
                    'score': 15,
                    'factors': [
                        'High labor costs',
                        'Inter-provincial trade barriers',
                        'Weather-related disruptions'
                    ]
                },
                'Technical_Risk': {
                    'score': 18,
                    'factors': [
                        'Strong digital infrastructure',
                        'Advanced cybersecurity measures',
                        'Growing tech ecosystem',
                        'High tech adoption rate'
                    ]
                }
            }
        }

    def calculate_risk_score(self, country):
        """Calculate comprehensive risk score for a country."""
        try:
            country_risks = self.risk_factors.get(country)
            if not country_risks:
                return None

            # Weights for different risk categories
            weights = {
                'Political_Risk': 0.25,
                'Economic_Risk': 0.30,
                'Operational_Risk': 0.20,
                'Technical_Risk': 0.25
            }

            # Calculate weighted risk score
            total_score = sum(weights[risk_type] * risk_data['score']
                            for risk_type, risk_data in country_risks.items())

            # Convert to risk rating
            if total_score <= 15:
                risk_rating = "Very Low Risk"
            elif total_score <= 25:
                risk_rating = "Low Risk"
            elif total_score <= 35:
                risk_rating = "Moderate Risk"
            elif total_score <= 45:
                risk_rating = "High Risk"
            else:
                risk_rating = "Very High Risk"

            return {
                'total_score': total_score,
                'risk_rating': risk_rating,
                'detailed_scores': {
                    risk_type: {
                        'score': risk_data['score'],
                        'factors': risk_data['factors']
                    }
                    for risk_type, risk_data in country_risks.items()
                }
            }

        except Exception as e:
            self.logger.error(f"Error calculating risk score for {country}: {str(e)}")
            return None

    def forecast_metrics(self, country, years_ahead=5):
        """Generate forecasts for key metrics."""
        try:
            if country not in self.historical_data:
                return None

            forecasts = {}
            metrics_to_forecast = ['GDP', 'Population', 'Consumer_Spending', 'Economic_Growth']

            for metric in metrics_to_forecast:
                historical_values = self.historical_data[country][metric]

                data = pd.Series(historical_values,
                               index=pd.date_range(start='2013', periods=len(historical_values), freq='Y'))

                model = ExponentialSmoothing(data, trend='add', seasonal=None)
                fitted_model = model.fit()
                forecast_values = fitted_model.forecast(years_ahead)

                current_value = data.iloc[-1]
                future_value = forecast_values.iloc[-1]
                growth_rate = ((future_value / current_value) - 1) * 100

                forecasts[metric] = {
                    'current_value': current_value,
                    'forecasted_value': future_value,
                    'growth_rate': growth_rate,
                    'trend': "Increasing" if growth_rate > 0 else "Decreasing"
                }

            return forecasts

        except Exception as e:
            self.logger.error(f"Error in forecasting for {country}: {str(e)}")
            return None

    def analyze_country(self, country):
        """Analyze a specific country's market potential with comprehensive risk assessment."""
        try:
            country_data = self.current_data.get(country)
            if not country_data:
                self.logger.error(f"No data found for country: {country}")
                return None

            market_score = country_data.get('Market_Score', 0)

            if market_score >= 88:
                status = "Highly Suitable"
            elif market_score >= 82:
                status = "Very Suitable"
            elif market_score >= 75:
                status = "Suitable"
            else:
                status = "Moderately Suitable"

            current_metrics = {
                'Population': f"{country_data['Population']:.1f} Million",
                'GDP': f"${country_data['GDP']:.2f} Trillion USD",
                'Consumer_Spending': f"${country_data['Consumer_Spending']:.2f} Trillion USD",
                'Economic_Growth': f"{country_data['Economic_Growth']:.1f}%"
            }

            forecasts = self.forecast_metrics(country)
            risk_analysis = self.calculate_risk_score(country)

            return {
                'country': country,
                'market_score': market_score,
                'status': status,
                'current_metrics': current_metrics,
                'strengths': country_data['Strengths'],
                'weaknesses': country_data['Weaknesses'],
                'forecasts': forecasts,
                'risk_analysis': risk_analysis
            }

        except Exception as e:
            self.logger.error(f"Error analyzing {country}: {str(e)}")
            return None

def create_comparison_visualizations(results):
    """Create comprehensive visualizations comparing different markets."""
    plt.figure(figsize=(15, 20))

    # Market Entry Scores Comparison
    plt.subplot(411)
    countries = [r['country'] for r in results]
    scores = [r['market_score'] for r in results]

    colors = ['#2ecc71' if s >= 88 else '#3498db' if s >= 82
              else '#f1c40f' if s >= 75 else '#e67e22' for s in scores]

    bars = plt.bar(range(len(countries)), scores, color=colors)
    plt.xticks(range(len(countries)), countries, rotation=45)
    plt.title('Market Entry Score Comparison', pad=20)
    plt.ylabel('Score (%)')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')

    # Overall Risk Score Comparison
    plt.subplot(412)
    risk_scores = [r['risk_analysis']['total_score'] for r in results]
    risk_colors = ['#2ecc71' if s <= 15 else '#3498db' if s <= 25
                   else '#f1c40f' if s <= 35 else '#e67e22' for s in risk_scores]

    risk_bars = plt.bar(range(len(countries)), risk_scores, color=risk_colors)
    plt.xticks(range(len(countries)), countries, rotation=45)
    plt.title('Overall Risk Score Comparison (Lower is Better)', pad=20)
    plt.ylabel('Risk Score')

    for bar in risk_bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom')

    # Detailed Risk Category Comparison
    plt.subplot(413)
    risk_categories = ['Political_Risk', 'Economic_Risk', 'Operational_Risk', 'Technical_Risk']
    x = np.arange(len(countries))
    width = 0.2

    for i, category in enumerate(risk_categories):
        category_scores = [r['risk_analysis']['detailed_scores'][category]['score']
                         for r in results]
        plt.bar(x + i*width - (len(risk_categories)-1)*width/2,
               category_scores, width, label=category.replace('_', ' '))

    plt.xticks(x, countries, rotation=45)
    plt.title('Risk Category Breakdown', pad=20)
    plt.ylabel('Risk Score')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Economic Metrics Comparison
    plt.subplot(414)
    metrics = ['GDP', 'Consumer_Spending', 'Economic_Growth']
    x = np.arange(len(countries))
    width = 0.25

    for i, metric in enumerate(metrics):
        values = [float(r['current_metrics'][metric].replace('$', '')
                       .replace(' Trillion USD', '')
                       .replace('%', ''))
                 for r in results]
        plt.bar(x + i*width - len(metrics)*width/2, values, width, label=metric)

    plt.xticks(x, countries, rotation=45)
    plt.title('Economic Metrics Comparison', pad=20)
    plt.ylabel('Value')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'market_comparison.png'), bbox_inches='tight', dpi=300)
    plt.close()

def analyze_markets(country_list):
    """Analyze multiple markets with comprehensive risk assessment."""
    analyzer = MarketEntryAnalyzer()
    print(f"\nAnalyzing countries: {', '.join(country_list)}")

    results = []
    for country in country_list:
        print(f"\nAnalyzing {country}...")
        analysis = analyzer.analyze_country(country)

        if analysis:
            results.append(analysis)

            print(f"\n{'-'*50}")
            print(f"Market Analysis: {country}")
            print(f"{'-'*50}")
            print(f"Market Entry Score: {analysis['market_score']:.2f}%")
            print(f"Status: {analysis['status']}")

            # Print risk analysis results
            risk_analysis = analysis['risk_analysis']
            print(f"\nRisk Assessment:")
            print(f"Overall Risk Rating: {risk_analysis['risk_rating']}")
            print(f"Total Risk Score: {risk_analysis['total_score']:.2f}")

            print("\nDetailed Risk Analysis:")
            for risk_type, risk_data in risk_analysis['detailed_scores'].items():
                print(f"\n{risk_type.replace('_', ' ')}:")
                print(f"Score: {risk_data['score']}")
                print("Key Factors:")
                for factor in risk_data['factors']:
                    print(f"- {factor}")

            print(f"\nCurrent Market Metrics:")
            for metric, value in analysis['current_metrics'].items():
                print(f"{metric}: {value}")

            print(f"\nKey Strengths:")
            for strength in analysis['strengths']:
                print(f"- {strength}")

            print(f"\nKey Challenges:")
            for weakness in analysis['weaknesses']:
                print(f"- {weakness}")

            if analysis['forecasts']:
                print(f"\n5-Year Market Outlook:")
                forecasts = analysis['forecasts']

                # GDP Forecast
                gdp = forecasts['GDP']
                print(f"GDP: ${gdp['forecasted_value']:.2f} Trillion USD ({gdp['trend']}, {gdp['growth_rate']:.1f}% growth)")

                # Population Forecast
                pop = forecasts['Population']
                print(f"Population: {pop['forecasted_value']:.1f} Million ({pop['trend']}, {pop['growth_rate']:.1f}% change)")

                # Consumer Spending Forecast
                spend = forecasts['Consumer_Spending']
                print(f"Consumer Spending: ${spend['forecasted_value']:.2f} Trillion USD ({spend['trend']}, {spend['growth_rate']:.1f}% growth)")

                # Economic Growth Forecast
                growth = forecasts['Economic_Growth']
                print(f"Expected GDP Growth Rate: {growth['forecasted_value']:.1f}%")

            print(f"{'-'*50}")

    if results:
        create_comparison_visualizations(results)
        print("\nVisualization saved as 'market_comparison.png'")

    return results

if __name__ == "__main__":
    test_countries = ["Japan", "Brazil", "France", "Canada"]
    results = analyze_markets(test_countries)
