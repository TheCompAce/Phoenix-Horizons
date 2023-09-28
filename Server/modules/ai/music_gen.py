from transformers import AutoProcessor, MusicgenForConditionalGeneration
import soundfile as sf
import argparse
import torch
from scipy.signal import hann  # Added for windowing function

def generate_audio(prompt, output_file, num_windows=3, window_size=1096, context_size=900, apply_smoothing=False):
    model_name = "facebook/musicgen-large"
    processor = AutoProcessor.from_pretrained(model_name)
    model = MusicgenForConditionalGeneration.from_pretrained(model_name)
    
    if torch.cuda.is_available():
        model = model.to('cuda')

    concatenated_audio = []
    
    inputs = processor(
        text=[prompt],
        padding=True,
        return_tensors="pt"
    )

    if torch.cuda.is_available():
        inputs = {key: val.to('cuda') for key, val in inputs.items()}

    audio_values = model.generate(**inputs, max_new_tokens=window_size)
    audio_np = audio_values[0].detach().cpu().numpy().reshape(-1)
    
    concatenated_audio.extend(audio_np)
  
    for _ in range(num_windows - 1):
        context = concatenated_audio[-context_size:]
        
        if apply_smoothing:
            window = hann(len(context))
            context = [c * w for c, w in zip(context, window)]
        
        context_tensor = torch.tensor(context, dtype=torch.float32)
        
        if torch.cuda.is_available():
            context_tensor = context_tensor.to('cuda')
        
        inputs = processor(
            text=[prompt],
            padding=True,
            return_tensors="pt"
        )
        
        if torch.cuda.is_available():
            inputs = {key: val.to('cuda') for key, val in inputs.items()}
        
        audio_values = model.generate(**inputs, max_new_tokens=window_size, do_sample=True, guidance_scale=2.0)
        
        # audio_np = audio_values[0].detach().cpu().numpy().reshape(-1)[context_size:]
        audio_np = audio_values[0].detach().cpu().numpy()
        
        if apply_smoothing:
            new_window = hann(len(audio_np))
            audio_np = [a * w for a, w in zip(audio_np, new_window)]
            
            blended_audio = [c + a for c, a in zip(context, audio_np[0:len(context)])]
            concatenated_audio[-context_size:] = blended_audio
        
        concatenated_audio.extend(audio_np)
    
    sf.write(output_file, concatenated_audio, 44100)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate music based on a text prompt.')
    parser.add_argument('prompt', type=str, help='Text prompt for music generation')
    parser.add_argument('output_file', type=str, help='Output file name for the generated music (.wav)')
    parser.add_argument('--num_windows', type=int, default=3, help='Number of sliding windows for a longer track')
    parser.add_argument('--window_size', type=int, default=1600, help='Size of each sliding window')
    parser.add_argument('--context_size', type=int, default=1000, help='Size of the context to keep for the next window')
    parser.add_argument('--apply_smoothing', type=bool, default=False, help='Apply smoothing for better track transition')
    
    args = parser.parse_args()
    
    generate_audio(args.prompt, args.output_file, args.num_windows, args.window_size, args.context_size, args.apply_smoothing)
