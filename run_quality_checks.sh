#!/bin/bash

# --- Script de Control de Calidad ---
# Este script ejecuta todas las comprobaciones de formateo y pruebas.
# Debe ejecutarse desde el directorio raíz del proyecto.

# Detener la ejecución si algún comando falla
set -e

echo "--- 1. Ordenando imports con isort ---"
isort .

echo "--- 2. Formateando código con black ---"
black .

echo "--- 3. Ejecutando pruebas unitarias con pytest ---"
pytest

echo ""
echo "✅ ¡Todas las comprobaciones de calidad han pasado con éxito! ✅"