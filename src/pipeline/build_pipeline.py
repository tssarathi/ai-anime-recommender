from src.etl.data_loader import AnimeDataLoader
from src.etl.vector_store import VectorStoreBuilder
from src.utils.custom_exception import CustomException
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    try:
        logger.info("Building Anime Recommendation Pipeline")
        data_loader = AnimeDataLoader(
            original_csv="data/bronze/anime_with_synopsis.csv",
            processed_csv="data/silver/anime_updated.csv",
        )
        processed_csv = data_loader.load_and_process()

        logger.info("Data loaded and processed successfully")

        vector_builder = VectorStoreBuilder(csv_path=processed_csv)
        vector_builder.build_and_save_vectorstore()

        logger.info("Vector store built successfully")

        logger.info("Anime Recommendation Pipeline Built Successfully")

    except Exception as e:
        logger.error(f"Error building Anime Recommendation Pipeline: {str(e)}")
        raise CustomException("Error building Anime Recommendation Pipeline", e)


if __name__ == "__main__":
    main()
