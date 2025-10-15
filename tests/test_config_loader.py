# tests/test_config_loader.py
from hdpo_gnn.utils.config_loader import load_configs
import os

def test_load_configs_successfully(tmp_path):
    """Verifica que la función carga y fusiona dos archivos YAML correctamente."""
    # 1. Arrange: Crear archivos de config falsos para la prueba
    setting_file = tmp_path / "setting.yml"
    setting_file.write_text("problem_params: {n_stores: 3}")
    
    hyperparams_file = tmp_path / "hyper.yml"
    hyperparams_file.write_text("optimizer_params: {lr: 0.001}")

    # 2. Act: Ejecutar la función que queremos probar
    config = load_configs(str(setting_file), str(hyperparams_file))

    # 3. Assert: Verificar que el resultado es el esperado
    assert isinstance(config, dict)
    assert config['problem_params']['n_stores'] == 3
    assert config['optimizer_params']['lr'] == 0.001