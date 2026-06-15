import pytest
from unittest.mock import patch, MagicMock
from main import run_ingestion, run_retrieval, main

@pytest.fixture
def mock_vector_store():
    """Fixture to provide a mocked VectorStore object."""
    return MagicMock()

@pytest.fixture
def mock_ingestion_object():
    """Fixture to provide a mocked DataIngestionPipeline object."""
    return MagicMock()

@patch("main.load_data.DataIngestionPipeline")
@patch("main.logger")
def test_run_ingestion_success(mock_logger, mock_pipeline_class, mock_vector_store):
    """Test successful initialization and execution of the data ingestion pipeline."""
    # Arrange
    mock_pipeline_instance = MagicMock()
    mock_pipeline_class.return_value = mock_pipeline_instance
    path = "dummy_data_path"

    # Act
    result = run_ingestion(path, mock_vector_store)

    # Assert
    mock_pipeline_class.assert_called_once_with(vector_store=mock_vector_store)
    # mock_pipeline_instance.run_pipeline.assert_called_once_with(path) # If uncommented in main
    mock_logger.info.assert_called_with("Data ingestion pipeline configured/completed successfully.")
    assert result == mock_pipeline_instance

@patch("main.load_data.DataIngestionPipeline")
@patch("main.logger")
def test_run_ingestion_failure(mock_logger, mock_pipeline_class, mock_vector_store):
    """Test ingestion pipeline failure handling and logging."""
    # Arrange
    mock_pipeline_class.side_effect = Exception("Ingestion Initialization Error")
    path = "dummy_data_path"

    # Act & Assert
    with pytest.raises(Exception, match="Ingestion Initialization Error"):
        run_ingestion(path, mock_vector_store)
    
    mock_logger.error.assert_called_once()

@patch("main.retrieve.RAGRetriever")
@patch("main.logger")
def test_run_retrieval_success(mock_logger, mock_retriever_class, mock_vector_store, mock_ingestion_object):
    """Test successful retrieval of an answer using RAGRetriever."""
    # Arrange
    mock_retriever_instance = MagicMock()
    mock_retriever_instance.get_answer.return_value = "Harry, Ron, and Hermione"
    mock_retriever_class.return_value = mock_retriever_instance
    query = "Who are the main characters?"

    # Act
    response = run_retrieval(query, mock_vector_store, mock_ingestion_object)

    # Assert
    mock_retriever_class.assert_called_once_with(
        vector_store=mock_vector_store, 
        ingestion_object=mock_ingestion_object
    )
    mock_retriever_instance.get_answer.assert_called_once_with(query)
    assert response == "Harry, Ron, and Hermione"

@patch("main.retrieve.RAGRetriever")
@patch("main.logger")
def test_run_retrieval_failure(mock_logger, mock_retriever_class, mock_vector_store, mock_ingestion_object):
    """Test retrieval failure handling and logging."""
    # Arrange
    mock_retriever_instance = MagicMock()
    mock_retriever_instance.get_answer.side_effect = Exception("Retrieval Error")
    mock_retriever_class.return_value = mock_retriever_instance
    query = "Who are the characters?"

    # Act & Assert
    with pytest.raises(Exception, match="Retrieval Error"):
        run_retrieval(query, mock_vector_store, mock_ingestion_object)
    
    mock_logger.error.assert_called_once()

@patch("main.run_retrieval")
@patch("main.run_ingestion")
@patch("main.vector_db.VectorStore")
@patch("main.logger")
def test_main_execution(mock_logger, mock_vector_store_class, mock_run_ingestion, mock_run_retrieval):
    """Test the main execution function orchestrates components correctly."""
    # Arrange
    mock_vector_store_instance = MagicMock()
    mock_vector_store_class.return_value = mock_vector_store_instance
    
    mock_ingestion_instance = MagicMock()
    mock_run_ingestion.return_value = mock_ingestion_instance
    
    mock_run_retrieval.return_value = "Success Response"
    
    pdf_path = "test_path"
    query = "test query"

    # Act
    result = main(pdf_path, query)

    # Assert
    mock_logger.info.assert_called_with("Starting Application...")
    mock_vector_store_class.assert_called_once()
    mock_run_ingestion.assert_called_once_with(pdf_path, mock_vector_store_instance)
    mock_run_retrieval.assert_called_once_with(query, mock_vector_store_instance, mock_ingestion_instance)
    assert result == "Success Response"
