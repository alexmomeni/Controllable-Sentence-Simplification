# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import fileinput
import sys

from access.preprocessors import get_preprocessors
from access.resources.prepare import prepare_models
from access.simplifiers import get_fairseq_simplifier, get_preprocessed_simplifier
from access.text import word_tokenize
from access.utils.helpers import yield_lines, write_lines, get_temp_filepath, mute

def transform(input_text, LengthRatioProcessor=0.95, LevenshteinPreprocessor=0.75, WordRankRatioPreprocessor=0.75, SentencePiecePreprocessor=10000):
    input_lines = input_text.split("\n")
    # Read from input
    source_filepath = get_temp_filepath()
    write_lines([word_tokenize(line) for line in input_lines], source_filepath)
    
    # Load best model
    best_model_dir = prepare_models()
    recommended_preprocessors_kwargs = {
        'LengthRatioPreprocessor': {'target_ratio': LengthRatioProcessor},
        'LevenshteinPreprocessor': {'target_ratio': LevenshteinPreprocessor},
        'WordRankRatioPreprocessor': {'target_ratio': WordRankRatioPreprocessor},
        'SentencePiecePreprocessor': {'vocab_size': SentencePiecePreprocessor},
    }
    preprocessors = get_preprocessors(recommended_preprocessors_kwargs)
    simplifier = get_fairseq_simplifier(best_model_dir,  beam=1)
    simplifier = get_preprocessed_simplifier(simplifier, preprocessors=preprocessors)
    # Simplify
    pred_filepath = get_temp_filepath()
    
    with mute():
        simplifier(source_filepath, pred_filepath)
    return list(yield_lines(pred_filepath))
    
