from unittest.mock import MagicMock, Mock


def test_predict(mock_heavy_dependencies):
    from ml import predict
    
    mock_torch, _ = mock_heavy_dependencies
    
    mock_inputs = MagicMock()
    mock_inputs.__contains__.return_value = True
    mock_inputs.__getitem__.return_value = mock_torch.tensor.return_value
    mock_inputs.to.return_value = mock_inputs
    
    mock_tokenizer = Mock(return_value=mock_inputs)
    mock_model = Mock(return_value={'logits': mock_torch.tensor.return_value})
    
    predictions = predict(mock_tokenizer, mock_model, 'cpu', ["test1", "test2"])
    assert predictions == [1, 1]
