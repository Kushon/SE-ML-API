# tests/conftest.py
import pytest
import sys
from unittest.mock import MagicMock, Mock

@pytest.fixture(scope="session", autouse=True)
def mock_heavy_dependencies():
    """Мокаем torch/transformers для ускорения тестов."""
    
    # Создаём моки
    mock_torch = MagicMock()
    mock_transformers = MagicMock()
    
    # torch.nn
    mock_nn = MagicMock()
    mock_torch.nn = mock_nn
    
    # Базовая настройка tensor
    mock_tensor = MagicMock()
    mock_tensor.squeeze.return_value = mock_tensor
    mock_tensor.item.return_value = 0.6
    mock_tensor.shape = MagicMock()
    
    mock_torch.tensor.return_value = mock_tensor
    mock_torch.ones.return_value = mock_tensor
    mock_torch.zeros.return_value = mock_tensor
    mock_torch.device.return_value = "cpu"
    
    # no_grad context manager
    mock_no_grad = MagicMock()
    mock_no_grad.__enter__.return_value = None
    mock_no_grad.__exit__.return_value = False
    mock_torch.no_grad.return_value = mock_no_grad
    
    # Transformers базовые классы
    mock_transformers.AutoTokenizer = Mock(return_value=Mock())
    mock_transformers.PreTrainedModel = Mock()
    mock_transformers.PretrainedConfig = Mock()
    
    # Патчим sys.modules
    sys.modules['torch'] = mock_torch
    sys.modules['torch.nn'] = mock_nn
    sys.modules['transformers'] = mock_transformers
    
    # Патчим ВСЕ подмодули torch/*
    for attr in dir(mock_torch):
        if attr.startswith('torch.'):
            sys.modules[attr] = mock_torch
    
    yield mock_torch, mock_transformers
