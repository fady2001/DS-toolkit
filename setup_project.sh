#!/bin/bash

# Create data folder structure with .gitkeep files
mkdir -p data/raw data/processed data/external data/interim

for folder in data/raw data/processed data/external data/interim; do
  if [ ! -f "$folder/.gitkeep" ]; then
    touch "$folder/.gitkeep"
  fi
done

echo "Data folder structure created with .gitkeep files."

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
  cat > .env << 'EOF'
# Environment variables go here, can be read by `python-dotenv` package:
#
#   `src/script.py`
#   ----------------------------------------------------------------
#    import dotenv
#
#    project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
#    dotenv_path = os.path.join(project_dir, '.env')
#    dotenv.load_dotenv(dotenv_path)
#   ----------------------------------------------------------------
#
# DO NOT ADD THIS FILE TO VERSION CONTROL!
EOF

  echo ".env file created with template content."
else
  echo ".env file already exists, skipping creation."
fi
