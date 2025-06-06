import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModel
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CarRecommendationSystem:
    """
    A comprehensive recommendation system for Mercedes-Benz that implements LLM integration capabilities.
    """

    def __init__(self, data_dir="./data", model_dir="./models"):
        """
        Initialize the recommendation system.

        Parameters:
        -----------
        data_dir : str
            Directory for storing data files
        model_dir : str
            Directory for storing trained models
        """
        self.vehicle_embeddings = None
        self.data_dir = data_dir
        self.model_dir = model_dir

        # Create directories if they don't exist
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        # Initialize components
        self.user_data = None
        self.vehicle_data = None
        self.interaction_data = None
        self.llm_model = None
        self.llm_tokenizer = None

        logger.info("Mercedes-Benz Recommendation System initialized")

    def generate_sample_data(self, num_users=100, num_vehicles=50, num_interactions=1000):
        """
        Generate synthetic sample data for testing the recommendation system.

        Parameters:
        -----------
        num_users : int
            Number of users to generate
        num_vehicles : int
            Number of vehicles to generate
        num_interactions : int
            Number of user-vehicle interactions to generate
        """
        logger.info(
            f"Generating sample data: {num_users} users, {num_vehicles} vehicles, {num_interactions} interactions")

        try:
            # Generate Users Data
            user_ids = [f"U{1000 + i}" for i in range(num_users)]
            locations = np.random.choice(["New York", "Berlin", "Tokyo", "London", "Dubai"],
                                         size=num_users, p=[0.3, 0.25, 0.2, 0.15, 0.1])

            self.user_data = pd.DataFrame({
                'user_id': user_ids,
                'age': np.random.randint(18, 75, size=num_users),
                'gender': np.random.choice(['Male', 'Female', 'Other'], size=num_users),
                'location': locations,
                'created_at': pd.date_range('2020-01-01', periods=num_users, freq='D')
            })

            # Generate Vehicles Data
            vehicle_ids = [f"V{100 + i}" for i in range(num_vehicles)]
            vehicle_types = np.random.choice(['Sedan', 'SUV', 'Coupe', 'Convertible', 'Electric'],
                                             size=num_vehicles, p=[0.3, 0.3, 0.15, 0.15, 0.1])

            self.vehicle_data = pd.DataFrame({
                'vehicle_id': vehicle_ids,
                'name': [f"Mercedes {t} {np.random.choice(['Class', 'Premium', 'Luxury'])}"
                         for t in vehicle_types],
                'type': vehicle_types,
                'price': np.clip(np.random.normal(50000, 15000, num_vehicles), 20000, 100000),
                'fuel_efficiency': np.random.uniform(15, 40, num_vehicles),
                'horsepower': np.random.randint(150, 600, num_vehicles),
                'safety_rating': np.random.uniform(3.5, 5.0, num_vehicles),
                'release_year': np.random.randint(2018, 2024, num_vehicles),
                'electric': np.where(vehicle_types == 'Electric', 1, 0)
            })

            # Generate Interactions Data
            interactions = []
            user_preferences = {
                'SUV': np.random.choice([3, 4, 5], size=num_users, p=[0.2, 0.3, 0.5]),
                'Sedan': np.random.choice([3, 4, 5], size=num_users, p=[0.3, 0.4, 0.3]),
                'Electric': np.random.choice([3, 4, 5], size=num_users, p=[0.5, 0.3, 0.2])
            }

            for _ in range(num_interactions):
                user_idx = np.random.randint(0, num_users)
                user_id = user_ids[user_idx]
                vehicle_idx = np.random.randint(0, num_vehicles)
                vehicle_id = vehicle_ids[vehicle_idx]
                vehicle_type = self.vehicle_data.iloc[vehicle_idx]['type']

                # Base rating with preference bias
                base_rating = np.random.normal(3.5, 0.5)
                if vehicle_type == 'SUV':
                    base_rating += user_preferences['SUV'][user_idx] / 5
                elif vehicle_type == 'Electric':
                    base_rating += user_preferences['Electric'][user_idx] / 5

                rating = np.clip(round(base_rating + np.random.normal(0, 0.3)), 1, 5)

                interactions.append({
                    'user_id': user_id,
                    'vehicle_id': vehicle_id,
                    'rating': rating,
                    'timestamp': pd.Timestamp.now() - pd.DateOffset(days=np.random.randint(0, 365))
                })

            self.interaction_data = pd.DataFrame(interactions)

            # Save generated data
            self.user_data.to_csv(os.path.join(self.data_dir, "users.csv"), index=False)
            self.vehicle_data.to_csv(os.path.join(self.data_dir, "vehicles.csv"), index=False)
            self.interaction_data.to_csv(os.path.join(self.data_dir, "interactions.csv"), index=False)

            logger.info(f"Sample data saved to {self.data_dir} directory")

        except Exception as e:
            logger.error(f"Error generating sample data: {str(e)}")
            raise

    def load_data(self, user_file, vehicle_file, interaction_file):
        """
        Load user, vehicle, and interaction data from files.

        Parameters:
        -----------
        user_file : str
            Path to user data CSV file
        vehicle_file : str
            Path to vehicle data CSV file
        interaction_file : str
            Path to user-vehicle interaction data CSV file
        """
        try:
            logger.info("Loading data from files...")
            self.user_data = pd.read_csv(os.path.join(self.data_dir, user_file))
            self.vehicle_data = pd.read_csv(os.path.join(self.data_dir, vehicle_file))
            self.interaction_data = pd.read_csv(os.path.join(self.data_dir, interaction_file))

            logger.info(f"Loaded {len(self.user_data)} users, {len(self.vehicle_data)} vehicles, "
                        f"and {len(self.interaction_data)} interactions")

            # Basic data validation
            if not {'user_id'}.issubset(set(self.user_data.columns)):
                raise ValueError("User data must contain 'user_id' column")

            if not {'vehicle_id'}.issubset(set(self.vehicle_data.columns)):
                raise ValueError("Vehicle data must contain 'vehicle_id' column")

            if not {'user_id', 'vehicle_id', 'rating'}.issubset(set(self.interaction_data.columns)):
                raise ValueError("Interaction data must contain 'user_id', 'vehicle_id', and 'rating' columns")

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    # Modified preprocess_data method in the MercedesRecommendationSystem class

    def preprocess_data(self):
        """
        Preprocess the loaded data for recommendation algorithms.
        """
        logger.info("Preprocessing data...")

        try:
            # Create a combined feature string for each vehicle for content-based filtering
            feature_cols = [col for col in self.vehicle_data.columns
                            if col not in ['vehicle_id', 'name', 'image_url']]

            # Initialize feature_string with empty strings
            self.vehicle_data['feature_string'] = ''

            for col in feature_cols:
                if self.vehicle_data[col].dtype != object:
                    # Handle numerical columns: include raw values
                    # Fill missing values with median
                    median_val = self.vehicle_data[col].median()
                    self.vehicle_data[col] = self.vehicle_data[col].fillna(median_val)
                    # Append numerical value as string with column name
                    self.vehicle_data['feature_string'] += ' ' + col + '_' + self.vehicle_data[col].round(2).astype(str)
                else:
                    # Handle categorical columns
                    # Fill missing values with 'unknown'
                    self.vehicle_data[col] = self.vehicle_data[col].fillna('unknown')
                    # Append category with column name
                    self.vehicle_data['feature_string'] += ' ' + col + '_' + self.vehicle_data[col].astype(str)

            # Normalize ratings for collaborative filtering
            min_rating = self.interaction_data['rating'].min()
            max_rating = self.interaction_data['rating'].max()
            if max_rating > min_rating:
                self.interaction_data['normalized_rating'] = ((self.interaction_data['rating'] - min_rating) /
                                                              (max_rating - min_rating)) * 5
            else:
                self.interaction_data['normalized_rating'] = 2.5

            logger.info("Data preprocessing completed")

        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise

    def preprocess_data__(self):
        """
        Preprocess the loaded data for recommendation algorithms.
        """
        logger.info("Preprocessing data...")

        try:
            # Create a combined feature string for each vehicle for content-based filtering
            feature_cols = [col for col in self.vehicle_data.columns
                            if col not in ['vehicle_id', 'name', 'image_url']]

            # Convert numerical features to strings and create a combined feature string
            self.vehicle_data['feature_string'] = ''
            for col in feature_cols:
                if self.vehicle_data[col].dtype != object:
                    # For numerical columns, categorize them with proper NaN handling
                    try:
                        # Create categories with explicit NaN handling
                        categorized = pd.qcut(
                            self.vehicle_data[col],
                            q=5,
                            labels=[f"{col}_very_low", f"{col}_low",
                                    f"{col}_medium", f"{col}_high",
                                    f"{col}_very_high"],
                            duplicates='drop'
                        )

                        # Add 'unknown' category explicitly
                        categorized = categorized.cat.add_categories('unknown')
                        categorized = categorized.fillna('unknown')

                        self.vehicle_data[f'{col}_cat'] = categorized
                        self.vehicle_data['feature_string'] += ' ' + self.vehicle_data[f'{col}_cat'].astype(str)

                    except Exception as e:
                        logger.warning(f"Could not categorize {col}: {str(e)}")
                        self.vehicle_data['feature_string'] += ' ' + self.vehicle_data[col].astype(str)
                else:
                    # For categorical columns, handle missing values first
                    self.vehicle_data[col] = self.vehicle_data[col].fillna('unknown')
                    self.vehicle_data['feature_string'] += ' ' + col + '_' + self.vehicle_data[col].astype(str)

            # Normalize ratings for collaborative filtering
            min_rating = self.interaction_data['rating'].min()
            max_rating = self.interaction_data['rating'].max()
            if max_rating > min_rating:  # Prevent division by zero
                self.interaction_data['normalized_rating'] = ((self.interaction_data['rating'] - min_rating) /
                                                              (max_rating - min_rating)) * 5
            else:
                self.interaction_data['normalized_rating'] = 2.5  # Midpoint if all ratings are equal

            logger.info("Data preprocessing completed")

        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise

    def initialize_llm(self, model_name="bert-base-uncased"):
        """
        Initialize a pretrained language model for text embedding.

        Parameters:
        -----------
        model_name : str
            Name of the pretrained model to load
        """
        logger.info(f"Initializing language model: {model_name}")

        try:
            self.llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.llm_model = AutoModel.from_pretrained(model_name)
            logger.info("Language model initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing language model: {str(e)}")
            raise

    def generate_text_embeddings(self, texts):
        """
        Generate embeddings for a list of texts using the LLM.

        Parameters:
        -----------
        texts : list
            List of text strings to encode

        Returns:
        --------
        numpy.ndarray
            Matrix of text embeddings
        """
        if self.llm_model is None or self.llm_tokenizer is None:
            logger.error("LLM not initialized. Call initialize_llm() first.")
            raise RuntimeError("LLM not initialized")

        try:
            # Make sure 'texts' is a list of strings
            if isinstance(texts, pd.Series):
                texts = texts.tolist()
            elif isinstance(texts, str):
                texts = [texts]

            # Tokenize and encode the texts
            encoded_input = self.llm_tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )

            # Generate embeddings
            with torch.no_grad():
                model_output = self.llm_model(**encoded_input)
                # For BERT-like models, use the [CLS] token's representation
                embeddings = model_output.last_hidden_state[:, 0, :]

            return embeddings.numpy()

        except Exception as e:
            logger.error(f"Error generating text embeddings: {str(e)}")
            raise

    def get_llm_enhanced_recommendations(self, user_query, top_n=5):
        """
        Use LLM to understand user query and enhance recommendations.

        Parameters:
        -----------
        user_query : str
            Natural language query from the user
        top_n : int
            Number of recommendations to return

        Returns:
        --------
        pandas.DataFrame
            DataFrame containing top_n recommended vehicles
        """
        if self.llm_model is None:
            logger.error("LLM not initialized. Call initialize_llm() first.")
            raise RuntimeError("LLM not initialized")

        logger.info(f"Getting LLM-enhanced recommendations for query: '{user_query}'")

        try:
            # Generate an embedding for the user query
            query_embedding = self.generate_text_embeddings([user_query])[0]

            # Generate embeddings for vehicle features if not already done
            if not hasattr(self, 'vehicle_embeddings'):
                logger.info("Generating vehicle embeddings...")
                # Convert pandas Series to list of strings
                feature_strings = self.vehicle_data['feature_string'].tolist()
                self.vehicle_embeddings = self.generate_text_embeddings(feature_strings)
                logger.info("Vehicle embeddings generated")

            # Calculate similarity between query and all vehicles
            similarities = cosine_similarity([query_embedding], self.vehicle_embeddings)[0]

            # Get indices of top matching vehicles
            top_indices = similarities.argsort()[::-1][:top_n]

            # Get the recommended vehicles
            recommended_vehicles = self.vehicle_data.iloc[top_indices].copy()
            recommended_vehicles['semantic_similarity'] = similarities[top_indices]

            logger.info(f"Found {len(recommended_vehicles)} LLM-enhanced recommendations")
            return recommended_vehicles

        except Exception as e:
            logger.error(f"Error getting LLM-enhanced recommendations: {str(e)}")
            raise

    def visualize_recommendations(self, recommendations, title="Recommendations", output_file=None):
        """
        Visualize recommendations.

        Parameters:
        -----------
        recommendations : pandas.DataFrame
            DataFrame containing recommendations
        title : str
            Title for the visualization
        output_file : str, optional
            Path to save the visualization image
        """
        try:
            plt.figure(figsize=(12, 8))

            if 'predicted_rating' in recommendations.columns:
                plt.barh(recommendations['name'], recommendations['predicted_rating'])
                plt.xlabel('Predicted Rating')
            elif 'similarity_score' in recommendations.columns:
                plt.barh(recommendations['name'], recommendations['similarity_score'])
                plt.xlabel('Similarity Score')
            elif 'hybrid_score' in recommendations.columns:
                plt.barh(recommendations['name'], recommendations['hybrid_score'])
                plt.xlabel('Hybrid Score')
            elif 'semantic_similarity' in recommendations.columns:
                plt.barh(recommendations['name'], recommendations['semantic_similarity'])
                plt.xlabel('Semantic Similarity')
            else:
                # Just list the recommendations without scores
                plt.barh(recommendations['name'], range(len(recommendations)))
                plt.xlabel('Rank')

            plt.ylabel('Vehicle')
            plt.title(title)
            plt.tight_layout()

            if output_file:
                plt.savefig(output_file)
                logger.info(f"Visualization saved to {output_file}")

            plt.show()

        except Exception as e:
            logger.error(f"Error visualizing recommendations: {str(e)}")
            raise

    def save_recommendations_to_file(self, recommendations, output_file):
        """
        Save recommendations to a CSV file.

        Parameters:
        -----------
        recommendations : pandas.DataFrame
            DataFrame containing recommendations
        output_file : str
            Path to save the recommendations
        """
        try:
            recommendations.to_csv(output_file, index=False)
            logger.info(f"Recommendations saved to {output_file}")

        except Exception as e:
            logger.error(f"Error saving recommendations: {str(e)}")
            raise

if __name__ == "__main__":
    """
    Example usage of the Mercedes-Benz Recommendation System.
    """
    # Initialize the recommendation system
    recsys = MercedesRecommendationSystem()

    # Generate sample data
    #recsys.generate_sample_data(num_users=500, num_vehicles=100, num_interactions=5000)

    try:
        # Load data
        recsys.load_data(
            user_file="users.csv",
            vehicle_file="vehicles.csv",
            interaction_file="interactions.csv"
        )

        # Preprocess data
        recsys.preprocess_data()

        # Initialize LLM for semantic understanding
        recsys.initialize_llm()

        # Get recommendations using different methods
        user_id = recsys.user_data['user_id'].iloc[0]
        vehicle_id = recsys.vehicle_data['vehicle_id'].iloc[0]

        # LLM-enhanced recommendations
        llm_recs = recsys.get_llm_enhanced_recommendations(
            "I want a convertible with horsepower more than 350, less than 30000"
        )
        recsys.visualize_recommendations(llm_recs, "LLM-Enhanced Recommendations")

        # Save the results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        recsys.save_recommendations_to_file(llm_recs, f"llm_{timestamp}.csv")
        logger.info("Example completed successfully")

    except Exception as e:
        logger.error(f"Error in example: {str(e)}")
        raise
