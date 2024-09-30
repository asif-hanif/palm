from .models.audio import get_audio_encoder
from .models.pengi import Projection

from .wrapper import PengiWrapper as Pengi

pengi = Pengi(config="base") 
pengi.args.classes_num = None
pengi.args.use_precomputed_melspec = False
pengi.args.pretrained_audioencoder_path = None


process_audio_fn = pengi.preprocess_audio
