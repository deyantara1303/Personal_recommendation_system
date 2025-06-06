import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import torch
from transformers import AutoTokenizer, AutoModel
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CarRecommendationSystem:
    """
    A comprehensive recommendation system for Mercedes-Benz that implements RAG-based recommendations.
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

    def preprocess_data(self):
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

    def create_vector_database(self):
        """
        Create a vector database for all vehicles in the system.
        """
        logger.info("Creating vector database for vehicles...")

        try:
            # Make sure we have a language model initialized
            if self.llm_model is None:
                self.initialize_llm(model_name="bert-base-uncased")

            # Create rich descriptions for each vehicle
            self.vehicle_data['rich_description'] = self.vehicle_data.apply(
                lambda x: f"Mercedes-Benz {x['name']} ({x['type']}): A {x['type'].lower()} "
                          f"with {x['horsepower']} horsepower, {x['fuel_efficiency']} MPG fuel efficiency, "
                          f"a safety rating of {x['safety_rating']}/5, "
                          f"released in {x['release_year']} with a price of ${int(x['price']):,}. "
                          f"{'This is an electric vehicle.' if x['electric'] == 1 else ''}",
                axis=1
            )

            # Generate embeddings for these descriptions
            descriptions = self.vehicle_data['rich_description'].tolist()
            self.vehicle_embeddings = self.generate_text_embeddings(descriptions)

            # Store the embeddings (could use FAISS, Pinecone, or other vector DB in production)
            self.vector_db = {
                'embeddings': self.vehicle_embeddings,
                'documents': descriptions,
                'vehicle_ids': self.vehicle_data['vehicle_id'].tolist()
            }

            logger.info("Vector database created successfully")

        except Exception as e:
            logger.error(f"Error creating vector database: {str(e)}")
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


    def rag_recommendation(self, user_query, top_k=5, user_id=None):
        """
        Generate recommendations using the RAG approach.

        Parameters:
        -----------
        user_query : str
            Natural language query from the user
        top_k : int
            Number of relevant vehicles to retrieve
        user_id : str, optional
            User ID for personalization

        Returns:
        --------
        dict
            Dictionary containing recommendations and explanations
        """
        logger.info(f"Processing RAG recommendation for query: '{user_query}'")

        try:
            # 1. Retrieve relevant vehicles based on query
            query_embedding = self.generate_text_embeddings([user_query])[0]
            similarities = cosine_similarity([query_embedding], self.vector_db['embeddings'])[0]

            # Get indices of top matching vehicles
            top_indices = similarities.argsort()[::-1][:top_k]
            top_similarities = similarities[top_indices]

            # Get the relevant vehicle information
            relevant_docs = [self.vector_db['documents'][i] for i in top_indices]
            relevant_vehicle_ids = [self.vector_db['vehicle_ids'][i] for i in top_indices]

            # 2. Format the retrieved information into a context
            context = "Based on the query, here are some relevant Mercedes-Benz vehicles:\n\n"
            for i, (doc, score) in enumerate(zip(relevant_docs, top_similarities)):
                context += f"{i + 1}. {doc} (Relevance Score: {score:.2f})\n\n"

            # 3. Add user preferences if available
            if user_id and user_id in self.user_data['user_id'].values:
                user_interactions = self.interaction_data[self.interaction_data['user_id'] == user_id]
                if not user_interactions.empty:
                    highest_rated = user_interactions.sort_values('rating', ascending=False).head(3)
                    liked_vehicles = self.vehicle_data[
                        self.vehicle_data['vehicle_id'].isin(highest_rated['vehicle_id'])]

                    context += "User preference information:\n"
                    for _, vehicle in liked_vehicles.iterrows():
                        context += f"- User has shown interest in {vehicle['name']} ({vehicle['type']})\n"

            # 4. Create a prompt for recommendation generation
            prompt = f"""
            You are a knowledgeable Mercedes-Benz vehicle recommendation assistant.

            USER QUERY: "{user_query}"

            AVAILABLE RELEVANT VEHICLES:
            {context}

            Based on the user's query and the available vehicle information:
            1. Analyze which aspects of the vehicles best match the user's requirements
            2. Identify any contradictions or tradeoffs in the user's requirements
            3. Recommend the most suitable vehicles
            4. Explain why each recommended vehicle matches the user's needs
            5. If there are tradeoffs being made, acknowledge them

            Format your response as a JSON with:
            - A "recommendations" list containing the top recommended vehicles
            - An "explanation" field with your detailed reasoning
            - A "tradeoffs" field discussing any compromises made
            """

            # 5. Future enhancement: Generate recommendations using an LLM API
            # For now, we'll create a simplified version of what an LLM would return

            # Get the recommended vehicles from the retrieved results
            recommended_vehicles = self.vehicle_data[self.vehicle_data['vehicle_id'].isin(relevant_vehicle_ids)].copy()
            recommended_vehicles['relevance_score'] = top_similarities

            # Sort by relevance score
            recommended_vehicles = recommended_vehicles.sort_values('relevance_score', ascending=False)

            # Create a manual explanation based on the query
            explanation = self._generate_explanation(user_query, recommended_vehicles)

            result = {
                "recommendations": recommended_vehicles.to_dict('records'),
                "explanation": explanation,
                "prompt": prompt  # Included for development; would remove in production
            }

            logger.info(f"Generated RAG recommendations successfully")
            return result

        except Exception as e:
            logger.error(f"Error generating RAG recommendations: {str(e)}")
            raise

    def _generate_explanation(self, query, recommended_vehicles):
        """
        Generate a simple explanation for recommendations.
        In production, this would be replaced with LLM-generated content.
        """
        query_lower = query.lower()

        # Extract key requirements
        requirements = {
            'convertible': 'convertible' in query_lower,
            'fuel_efficient': any(term in query_lower for term in ['fuel', 'efficient', 'economy', 'mpg']),
            'cheap': any(term in query_lower for term in ['cheap', 'inexpensive', 'affordable', 'low price']),
            'safety': any(term in query_lower for term in ['safe', 'safety']),
            'powerful': any(term in query_lower for term in ['power', 'horsepower', 'hp', 'fast'])
        }

        # Check for numeric requirements
        import re
        hp_match = re.search(r'horsepower (?:more than|greater than|over|above) (\d+)', query_lower)
        min_hp = int(hp_match.group(1)) if hp_match else 0

        # Generate explanation
        explanation = "Based on your requirements, I've found these Mercedes-Benz vehicles that best match your needs.\n\n"

        # Identify tradeoffs
        tradeoffs = []
        if requirements['powerful'] and requirements['fuel_efficient']:
            tradeoffs.append("There's typically a tradeoff between high horsepower and fuel efficiency.")
        if requirements['powerful'] and requirements['cheap']:
            tradeoffs.append("High-performance vehicles tend to come at a higher price point.")

        if tradeoffs:
            explanation += "Note that there are some tradeoffs to consider:\n"
            for tradeoff in tradeoffs:
                explanation += f"- {tradeoff}\n"
            explanation += "\n"

        # Add vehicle-specific explanations
        for i, vehicle in recommended_vehicles.head(3).iterrows():
            explanation += f"• {vehicle['name']} ({vehicle['type']}):\n"

            matches = []
            if requirements['convertible'] and vehicle['type'] == 'Convertible':
                matches.append("is a convertible as requested")

            if requirements['fuel_efficient'] and vehicle['fuel_efficiency'] > 25:
                matches.append(f"offers good fuel efficiency at {vehicle['fuel_efficiency']:.1f} MPG")

            if requirements['cheap'] and vehicle['price'] < 50000:
                matches.append(f"is relatively affordable at ${int(vehicle['price']):,}")

            if requirements['safety'] and vehicle['safety_rating'] > 4:
                matches.append(f"has excellent safety features with a {vehicle['safety_rating']:.1f}/5 rating")

            if requirements['powerful'] and vehicle['horsepower'] > 300:
                matches.append(f"provides strong performance with {vehicle['horsepower']} horsepower")

            if min_hp > 0 and vehicle['horsepower'] > min_hp:
                matches.append(f"exceeds your minimum horsepower requirement of {min_hp}")

            if matches:
                explanation += "  This vehicle " + ", ".join(matches) + ".\n\n"

        return explanation

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
        Example usage of the Mercedes-Benz Recommendation System with RAG.
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

            # Create vector database for RAG
            recsys.create_vector_database()

            # Process a natural language query using RAG
            query = "I want a convertible"
            rag_results = recsys.rag_recommendation(query, top_k=5)

            # Print the results
            print("\n=== RAG Recommendation Results ===")
            print("\nExplanation:")
            print(rag_results["explanation"])

            print("\nTop Recommended Vehicles:")
            for i, vehicle in enumerate(rag_results["recommendations"][:3]):
                print(f"{i + 1}. {vehicle['name']} ({vehicle['type']})")
                print(f"   - Horsepower: {vehicle['horsepower']}")
                print(f"   - Fuel Efficiency: {vehicle['fuel_efficiency']:.1f} MPG")
                print(f"   - Price: ${int(vehicle['price']):,}")
                print(f"   - Safety Rating: {vehicle['safety_rating']:.1f}/5")
                print(f"   - Relevance Score: {vehicle['relevance_score']:.2f}")
                print()

            logger.info("RAG example completed successfully")

        except Exception as e:
            logger.error(f"Error in RAG example: {str(e)}")
            raise