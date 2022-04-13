import tensorflow as tf
import numpy as np
import pickle
import pandas as pd
from shutil import copy2
from datetime import datetime
import os
import note_seq

import sys

sys.path.insert(0, "../")

from hvo_sequence.hvo_sequence.io_helpers import note_sequence_to_hvo_sequence
from hvo_sequence.hvo_sequence.drum_mappings import ROLAND_REDUCED_MAPPING

tf.compat.v1.enable_eager_execution()

# For some reason, tfds import gives an error on the first attempt but works second time around
try:
    import tensorflow_datasets as tfds
except:
    import tensorflow_datasets as tfds


# Import magenta's note_seq
dataset_train_unprocessed, dataset_train_info = tfds.load(
    name="groove/2bar-midionly", split=tfds.Split.TRAIN, try_gcs=True, with_info=True
)

dataset_test_unprocessed = tfds.load(
    name="groove/2bar-midionly", split=tfds.Split.TEST, try_gcs=True
)

dataset_validation_unprocessed = tfds.load(
    name="groove/2bar-midionly", split=tfds.Split.VALIDATION, try_gcs=True
)

# In all three sets, separate entries into individual examples
dataset_train = dataset_train_unprocessed.batch(1)
dataset_test = dataset_test_unprocessed.batch(1)
dataset_validation = dataset_validation_unprocessed.batch(1)

print(
    "\n Number of Examples in Train Set: {}, Test Set: {}, Validation Set: {}".format(
        len(list(dataset_train)), len(list(dataset_test)), len(list(dataset_validation))
    )
)

dataframe = pd.read_csv("resources/source_dataset/groove/info.csv", delimiter=",")


def dict_append(dictionary, key, vals):
    # Appends a value or a list of values to a key in a dictionary

    # if the values for a key are not a list, they are converted to a list and then extended with vals
    dictionary[key] = (
        list(dictionary[key])
        if not isinstance(dictionary[key], list)
        else dictionary[key]
    )

    # if vals is a single value (not a list), it's converted to a list so as to be iterable
    vals = [vals] if not isinstance(vals, list) else vals

    # append new values
    for val in vals:
        dictionary[key].append(val)

    return dictionary


def convert_groove_midi_dataset(
    dataset, beat_division_factors=[4], csv_dataframe_info=None
):

    dataset_dict_processed = dict()
    dataset_dict_processed.update(
        {
            "drummer": [],
            "session": [],
            "loop_id": [],  # the id of the recording from which the loop is extracted
            "master_id": [],  # the id of the recording from which the loop is extracted
            "style_primary": [],
            "style_secondary": [],
            "bpm": [],
            "beat_type": [],
            "time_signature": [],
            "full_midi_filename": [],
            "full_audio_filename": [],
            "midi": [],
            "note_sequence": [],
            "hvo_sequence": [],
        }
    )

    for features in dataset:

        # Features to be extracted from the dataset

        note_sequence = note_seq.midi_to_note_sequence(
            tfds.as_numpy(features["midi"][0])
        )

        # ignore if no notes in note_sequence (i.e. empty 2 bar sequence)
        if note_sequence.notes:

            _hvo_seq = note_sequence_to_hvo_sequence(
                ns=note_sequence,
                drum_mapping=ROLAND_REDUCED_MAPPING,
                beat_division_factors=beat_division_factors,
            )

            if (
                (not csv_dataframe_info.empty)
                and len(_hvo_seq.time_signatures) == 1
                and len(_hvo_seq.tempos) == 1
            ):

                # Master ID for the Loop
                main_id = features["id"].numpy()[0].decode("utf-8").split(":")[0]

                # Get the relevant series from the dataframe
                df = csv_dataframe_info[csv_dataframe_info.id == main_id]

                # Update the dictionary associated with the metadata
                dict_append(
                    dataset_dict_processed, "drummer", df["drummer"].to_numpy()[0]
                )
                _hvo_seq.metadata.drummer = df["drummer"].to_numpy()[0]

                dict_append(
                    dataset_dict_processed,
                    "session",
                    df["session"].to_numpy()[0].split("/")[-1],
                )
                _hvo_seq.metadata.session = df["session"].to_numpy()[0]

                dict_append(
                    dataset_dict_processed,
                    "loop_id",
                    features["id"].numpy()[0].decode("utf-8"),
                )
                _hvo_seq.metadata.loop_id = features["id"].numpy()[0].decode("utf-8")

                dict_append(dataset_dict_processed, "master_id", main_id)
                _hvo_seq.metadata.master_id = main_id

                style_full = df["style"].to_numpy()[0]
                style_primary = style_full.split("/")[0]

                dict_append(dataset_dict_processed, "style_primary", style_primary)
                _hvo_seq.metadata.style_primary = style_primary

                if "/" in style_full:
                    style_secondary = style_full.split("/")[1]
                    dict_append(
                        dataset_dict_processed, "style_secondary", style_secondary
                    )
                    _hvo_seq.metadata.style_secondary = style_secondary
                else:
                    dict_append(dataset_dict_processed, "style_secondary", ["None"])
                    _hvo_seq.metadata.style_secondary = "None"

                dict_append(dataset_dict_processed, "bpm", df["bpm"].to_numpy()[0])

                dict_append(
                    dataset_dict_processed, "beat_type", df["beat_type"].to_numpy()[0]
                )
                _hvo_seq.metadata.beat_type = df["beat_type"].to_numpy()[0]

                dict_append(
                    dataset_dict_processed,
                    "time_signature",
                    df["time_signature"].to_numpy()[0],
                )

                dict_append(
                    dataset_dict_processed,
                    "full_midi_filename",
                    df["midi_filename"].to_numpy()[0],
                )
                _hvo_seq.metadata.full_midi_filename = df["midi_filename"].to_numpy()[0]

                dict_append(
                    dataset_dict_processed,
                    "full_audio_filename",
                    df["audio_filename"].to_numpy()[0],
                )
                _hvo_seq.metadata.full_audio_filename = df["audio_filename"].to_numpy()[
                    0
                ]

                dict_append(dataset_dict_processed, "midi", features["midi"].numpy()[0])
                dict_append(dataset_dict_processed, "note_sequence", [note_sequence])

                dict_append(dataset_dict_processed, "hvo_sequence", _hvo_seq)

        else:
            pass

    return dataset_dict_processed


# Process Training Set
dataset_train = dataset_train_unprocessed.batch(1)
dataset_train_processed = convert_groove_midi_dataset(
    dataset=dataset_train, beat_division_factors=[4], csv_dataframe_info=dataframe
)

# Process Test Set
dataset_test = dataset_test_unprocessed.batch(1)
dataset_test_processed = convert_groove_midi_dataset(
    dataset=dataset_test, beat_division_factors=[4], csv_dataframe_info=dataframe
)

# Process Validation Set
dataset_validation = dataset_validation_unprocessed.batch(1)
dataset_validation_processed = convert_groove_midi_dataset(
    dataset=dataset_validation, beat_division_factors=[4], csv_dataframe_info=dataframe
)


def sort_dictionary_by_key(dictionary_to_sort, key_used_to_sort):
    # sorts a dictionary according to the list within a given key
    sorted_ids = np.argsort(dictionary_to_sort[key_used_to_sort])
    for key in dictionary_to_sort.keys():
        dictionary_to_sort[key] = [dictionary_to_sort[key][i] for i in sorted_ids]
    return dictionary_to_sort


# Sort the sets using ids
dataset_train_processed = sort_dictionary_by_key(dataset_train_processed, "loop_id")
dataset_test_processed = sort_dictionary_by_key(dataset_test_processed, "loop_id")
dataset_validation_processed = sort_dictionary_by_key(
    dataset_validation_processed, "loop_id"
)


# DUMP INTO A PICKLE FILE
def store_dataset_as_pickle(
    dataset_list,
    filename_list,
    root_dir="processed_dataset",
    append_datetime=True,
    features_with_separate_picklefile=["hvo", "midi", "note_seq"],
):

    # filename = filename.split(".obj")[0]
    path = root_dir

    if append_datetime:
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y_at_%H_%M_hrs")
    else:
        dt_string = ""

    path = os.path.join(path, "Processed_On_" + dt_string)

    if not os.path.exists(path):
        os.makedirs(path)

    # copy2(os.path.join(os.getcwd(), currentNotebook), path)

    for i, dataset in enumerate(dataset_list):

        subdirectory = os.path.join(path, filename_list[i])
        if not os.path.exists(subdirectory):
            os.makedirs(subdirectory)

        print("-" * 100)
        print("-" * 100)
        print("Processing %s folder" % subdirectory)
        print("-" * 100)
        print("-" * 100)

        # Create Metadata File
        csv_dataframe = pd.DataFrame()

        for k in dataset.keys():
            if k not in features_with_separate_picklefile:
                csv_dataframe[k] = dataset[k]

        csv_dataframe.to_csv(os.path.join(subdirectory, "metadata.csv"))

        print("Metadata created!")
        print("-" * 100)

        for feature in features_with_separate_picklefile:
            if feature in dataset.keys():
                dataset_filehandler = open(
                    os.path.join(subdirectory, "%s_data.obj" % feature), "wb"
                )
                print(feature)
                print(dataset_filehandler)
                pickle.dump(dataset[feature], dataset_filehandler)
                dataset_filehandler.close()
                print(
                    "feature %s pickled at %s"
                    % (
                        feature,
                        os.path.join(
                            subdirectory, "%s.obj" % filename_list[i].split(".")[0]
                        ),
                    )
                )
                print("-" * 100)

            else:
                raise Warning("Feature is not available: ", feature)


dataset_list = [
    dataset_train_processed,
    dataset_test_processed,
    dataset_validation_processed,
]

filename_list = [
    "GrooveMIDI_processed_train",
    "GrooveMIDI_processed_test",
    "GrooveMIDI_processed_validation",
]

store_dataset_as_pickle(
    dataset_list,
    filename_list,
    root_dir="processed_dataset",
    append_datetime=True,
    features_with_separate_picklefile=["hvo_sequence", "midi", "note_sequence"],
)

dataset_list = [
    dataset_train_processed,
    dataset_test_processed,
    dataset_validation_processed,
]

filename_list = [
    "GrooveMIDI_processed_train",
    "GrooveMIDI_processed_test",
    "GrooveMIDI_processed_validation",
]

store_dataset_as_pickle(
    dataset_list,
    filename_list,
    root_dir="processed_dataset",
    append_datetime=True,
    features_with_separate_picklefile=["hvo_sequence", "midi", "note_sequence"],
)
