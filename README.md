# GMD2HVO_PreProcessing
---------------
### Accessing Full Performances
GrooveMIDI (GMD) dataset is available in two ways.

   1. Groove-v1.0.0.zip available in https://magenta.tensorflow.org/datasets/groove. 
   This zip file contains the full performances without any processing. 
   Moreover, the metadata is available here via the info.csv file. 
   
   2. Alternatively, GMD is accessible via tensorflow datasets (tfds)
   
    import tensorflow as tf
    import tensorflow_datasets as tfds
        
    # tfds works in both Eager and Graph modes
    tf.enable_eager_execution()
        
    # Load the full GMD with MIDI only (no audio) as a tf.data.Dataset
    dataset = tfds.load(
        name="groove/full-midionly",
        split=tfds.Split.TRAIN,
        try_gcs=True)

---------------
### Accessing 2Bar/4Bar Segments of Performances

The dataset is also preprocessed into 2-bar and 4-bar segments. 
However, (to best of my knowledge) these are only accessible via tfds

    dataset = tfds.load(
        name="groove/2bar-midionly",        # OR, "groove/4bar-midionly"
        split=tfds.Split.TRAIN,
        try_gcs=True)

<b><u>NOTE:</u></b> In either case, the metadata should be accessed via  <b> info.csv in Groove-v1.0.0.zip</b> 

-------
### What Does This Repo Do?

Using the Repo, grab the 2-bar or 4-bar segments, convert them into HVO_Sequences 
and embed the metadata (accessed from info.csv) in the HVO_Sequences. 
Finally, for each train/test/validation subsets of data (as already split by Magenta), 
we store the processed data in a number of pickle and csv files

    1. hvo_sequence_data.obj  -->  pickled list of loops processed into HVO_Sequence format
    2. midi_data.obj          -->  pickled list of  loops provided by Magenta in  pretty_midi format
    3. note_sequence_data.obj -->  pickled list of  loops provided by Magenta in  note_sequence format
    4. metadata.csv           --> corresponding metadata
    
<b><u>NOTE:</u></b> All the indices in the lists above as well as the rows in the metadata correspond to each other. 


-------
### Repository Structure Pre/Post Processing

![guide](https://user-images.githubusercontent.com/35939495/118392795-aacfc180-b63b-11eb-9ae7-25493a08cb7f.jpg)
