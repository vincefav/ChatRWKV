########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os
import numpy as np
from rwkv.model import RWKV
from rwkv.utils import PIPELINE

# Ensure the environment variables are set for GPU usage
os.environ['RWKV_JIT_ON'] = '1'
os.environ["RWKV_CUDA_ON"] = '1'  # Enable CUDA for GPU execution
os.environ["RWKV_RESCALE_LAYER"] = '999'

# Adjust the model and tokenizer file paths as necessary
MODEL_FILE = '/content/drive/MyDrive/AI/RWKV/RWKV-4-MIDI-560M-v1-20230717-ctx4096.pth'
TOKENIZER_FILE = "/content/MIDI-LLM-tokenizer/tokenizer-midi.json"

# Initialize the model for GPU execution
model = RWKV(model=MODEL_FILE, strategy='cuda fp16')
pipeline = PIPELINE(model, TOKENIZER_FILE)

# Determine if the model is for ABC or MIDI based on the file name
ABC_MODE = ('-ABC-' in MODEL_FILE)
MIDI_MODE = ('-MIDI-' in MODEL_FILE)

# For MIDI_MODE, use the pipeline directly as the tokenizer
if MIDI_MODE:
    tokenizer = pipeline
    EOS_ID = 0
    TOKEN_SEP = ' '

##########################################################################################################
#
# MIDI model:
# Use https://github.com/briansemrau/MIDI-LLM-tokenizer/blob/main/str_to_midi.py to convert output to MIDI
# Use https://midiplayer.ehubsoft.net/ and select Full MIDI Player (30M) to play MIDI
# For best results: install https://coolsoft.altervista.org/en/virtualmidisynth
# and use soundfont: https://musical-artifacts.com/artifacts/1720
# 
# ABC model:
# Use https://abc.rectanglered.com to play it
# dataset: load_dataset("sander-wood/massive_abcnotation_dataset")["train"]
# Our training data format: [2] + idx + [3], where idx starts with "control code" (https://huggingface.co/datasets/sander-wood/massive_abcnotation_dataset)
#
##########################################################################################################

def generate_unconditioned_midi(trial=1):
    for TRIAL in range(trial):
        print(f"Generating unconditioned MIDI #{TRIAL}")
        ccc = '<pad>'  # Start with minimal pre-conditions
        ccc_output = '<start>'
        fout = open(f"unconditioned_midi_{TRIAL}.txt", "w")
        fout.write(ccc_output)

        generate_midi_sequence(ccc, fout)
        fout.write(' <end>')
        fout.close()

def generate_continuation_from_piano_midi(input_sequence, trial=1):
    for TRIAL in range(trial):
        print(f"Generating continuation for MIDI #{TRIAL}")
        ccc = '<pad> ' + input_sequence  # Pre-condition with input sequence
        ccc_output = '<start> ' + input_sequence
        fout = open(f"continued_midi_{TRIAL}.txt", "w")
        fout.write(ccc_output)

        generate_midi_sequence(ccc, fout)
        fout.write(' <end>')
        fout.close()

def generate_midi_sequence(ccc, fout):
    occurrence = {}
    state = None
    for i in range(4096):
        if i == 0:
            out, state = model.forward(tokenizer.encode(ccc), state)
        else:
            out, state = model.forward([token], state)

        if MIDI_MODE:  # Specific adjustments for MIDI mode
            # Generate the token before calling adjust_midi_output
            token = pipeline.sample_logits(out, temperature=1.0, top_k=8, top_p=0.8)
            
            # Now call adjust_midi_output with the token as a parameter
            adjust_midi_output(out, occurrence, i, token)
        
        if token == EOS_ID: break

        fout.write(TOKEN_SEP + tokenizer.decode([token]))
        fout.flush()

def adjust_midi_output(out, occurrence, i, token):
    for n in occurrence:
        out[n] -= (0 + occurrence[n] * 0.5)
    out[0] += (i - 2000) / 500  # Adjust length bias
    out[127] -= 1  # Avoid specific tokens if necessary
    for n in occurrence: occurrence[n] *= 0.997  # Decay repetition penalty
    if token >= 128 or token == 127:
        occurrence[token] = 1 + (occurrence[token] if token in occurrence else 0)
    else:
        occurrence[token] = 0.3 + (occurrence[token] if token in occurrence else 0)
