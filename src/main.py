import os
from .pipeline import TextPipeline
from .data_loader import load_data

if __name__ == '__main__':
    try:
        # Obtener el directorio raíz del proyecto
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        file_path = os.path.join(root_dir, 'data', 'training_set_rel3.tsv')
        
        data = load_data(file_path)
        pipeline = TextPipeline(data)
        pipeline.run_pipeline()
        
    except FileNotFoundError as e:
        print(f"Error: Archivo no encontrado - {e}")
    except ValueError as e:
        print(f"Error de valor: {e}")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")
