from src.config.config import GROQ_API_KEY, MODEL_NAME
from src.etl.vector_store import VectorStoreBuilder
from src.llm.recommender import AnimeRecommender
from src.utils.custom_exception import CustomException
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AnimeRecommendationPipeline:
    def __init__(self, persist_dir: str = "data/gold/"):
        try:
            logger.info("Initializing Anime Recommendation Pipeline")
            vector_builder = VectorStoreBuilder(
                csv_path="/../data/silver/anime_updated.csv", persist_dir=persist_dir
            )

            retriever = vector_builder.load_vector_store().as_retriever(
                search_kwargs={"k": 10}
            )
            self.recommender = AnimeRecommender(retriever, GROQ_API_KEY, MODEL_NAME)

            logger.info("Anime Recommendation Pipeline Initialized Successfully")

        except Exception as e:
            logger.error(
                f"Anime Recommendation Pipeline Initialization Failed: {str(e)}"
            )
            raise CustomException("Error during pipleine initialization", e)

    def recommend(self, query: str) -> tuple[str, list]:
        try:
            logger.info(f"Recommendation requested for query: {query}")
            recommendations, source_docs = self.recommender.get_recommendations(query)
            logger.info(f"Recommendations: {recommendations}")
            return recommendations, source_docs
        except Exception as e:
            logger.error(f"Error during recommendation: {str(e)}")
            raise CustomException("Error during recommendation", e)
