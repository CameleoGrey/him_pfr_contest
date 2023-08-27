
import os
from pathlib import Path

configs_dir = os.path.join("..", "..", "configs")
if not Path( configs_dir ).exists():
    Path( configs_dir ).mkdir(parents=True, exist_ok=True)

models_dir = os.path.join("..", "..", "models")
if not Path( models_dir ).exists():
    Path( models_dir ).mkdir(parents=True, exist_ok=True)

data_dir = os.path.join("..", "..", "data")
if not Path( data_dir ).exists():
    Path( data_dir ).mkdir(parents=True, exist_ok=True)

raw_dir = os.path.join( data_dir, "raw" )
if not Path( raw_dir ).exists():
    Path( raw_dir ).mkdir(parents=True, exist_ok=True)

log_dir = os.path.join( data_dir, "logs" )
if not Path( log_dir ).exists():
    Path( log_dir ).mkdir(parents=True, exist_ok=True)

images_dir = os.path.join( raw_dir, "images" )
if not Path( images_dir ).exists():
    Path( images_dir ).mkdir(parents=True, exist_ok=True)

interim_dir = os.path.join( data_dir, "interim" )
if not Path( interim_dir ).exists():
    Path( interim_dir ).mkdir(parents=True, exist_ok=True)

result_dir = os.path.join( data_dir, "result" )
if not Path( result_dir ).exists():
    Path( result_dir ).mkdir(parents=True, exist_ok=True)
