import numpy as np
import note_seq
from note_seq.protobuf import music_pb2
import soundfile as sf
from bokeh.plotting import output_file, show, save
from bokeh.models import Span
import warnings

from hvo_sequence.utils import is_power_of_two, create_grid_for_n_bars, find_pitch_and_tag
from hvo_sequence.custom_dtypes import Tempo, Time_Signature

class HVO_Sequence(object):

    def __init__(self, drum_mapping=None):
        """

        @param drum_mapping:                        a dictionary of grouped midi notes and corresponding names
                                                    {..., "Snare": [38, 37, 40], ...}

        Note:                                       A user can only manually modify the following parameters:

                                                        time signature
                                                        tempo
                                                        beat division factors
                                                        pitch class groups and tags
                                                        hvo array of (time_steps, 3*n_voices)
        """

        # Create ESSENTIAL fields used for the sequence
        # NOTE: DO NOT MODIFY THE BELOW MANGLED VARIABLES DIRECTLY
        #       RATHER USE THE PROPERTY GETTER AND SETTERS TO ENSURE
        #       DATATYPE CONSISTENCY
        self.__time_signatures = list()
        self.__tempos = list()
        self.__drum_mapping = None
        self.__hvo = None

        # Use property setters to initiate properties (DON"T ASSIGN ABOVE so that the correct datatype is checked)
        if drum_mapping:
            self.drum_mapping = drum_mapping

    #   ----------------------------------------------------------------------
    #   Property getters and setter wrappers for ESSENTIAL class variables
    #   ----------------------------------------------------------------------

    @property
    def time_signatures(self):
        return self.__time_signatures

    def add_time_signature(self, time_step=None, numerator=None, denominator=None, beat_division_factors=None):
        time_signature = Time_Signature(time_step=time_step, numerator=numerator,
                                        denominator=denominator, beat_division_factors=beat_division_factors)
        self.time_signatures.append(time_signature)
        return time_signature

    @property
    def tempos(self):
        return self.__tempos

    def add_tempo(self, time_step=None, qpm=None):
        tempo = Tempo(time_step=time_step, qpm=qpm)
        self.tempos.append(tempo)
        return tempo

    @property
    def drum_mapping(self):
        if not self.__drum_mapping:
            warnings.warn("drum_mapping is not specified")
        return self.__drum_mapping

    @drum_mapping.setter
    def drum_mapping(self, drum_map):
        # Ensure drum map is a dictionary
        assert isinstance(drum_map, dict), "drum_mapping should be a dict" \
                                           "of {'Drum Voice Tag': [midi numbers]}"

        # Ensure the values in each key are non-empty list of ints between 0 and 127
        for key in drum_map.keys():
            assert isinstance(drum_map[key], list), "map[{}] should be a list of MIDI numbers " \
                                                    "(int between 0-127)".format(drum_map[key])
            if len(drum_map[key]) >= 1:
                assert all([isinstance(val, int) for val in drum_map[key]]), "Expected list of ints in " \
                                                                             "map[{}]".format(drum_map[key])
            else:
                assert False, "map[{}] is empty --> should be a list of MIDI numbers " \
                              "(int between 0-127)".format(drum_map[key])

        if self.hvo is not None:
            assert self.hvo.shape[1] % len(drum_map.keys()) == 0, \
                "The second dimension of hvo should be three times the number of drum voices, len(drum_mapping.keys())"

        # Now, safe to update the local drum_mapping variable
        self.__drum_mapping = drum_map

    @property
    def hvo(self):
        return self.__hvo

    @hvo.setter
    def hvo(self, x):
        # Ensure x is a numpy.ndarray of shape (number of steps, 3*number of drum voices)
        assert isinstance(x, np.ndarray), "Expected numpy.ndarray of shape (time_steps, 3 * number of voices), " \
                                          "but received {}".format(type(x))
        # if drum_mapping already exists, the dimension at each time-step should be 3 times
        #   the number of voices in the drum_mapping
        if self.drum_mapping:
            assert x.shape[1] / len(self.drum_mapping.keys()) == 3, \
                "The second dimension of hvo should be three times the number of drum voices, len(drum_mapping.keys())"

        # Now, safe to update the local hvo score array
        self.__hvo = x

    #   --------------------------------------------------------------
    #   Useful properties calculated from ESSENTIAL class variables
    #   --------------------------------------------------------------

    @property
    def number_of_voices(self):
        calculable = self.is_drum_mapping_available(print_missing=True)
        if not calculable:
            print("can't calculate the number of voices as the drum_mapping is missing")
        else:
            return int(self.hvo.shape[1] / 3)

    @property
    def hits(self):
        calculable = all([self.is_hvo_score_available(print_missing=True),
                          self.is_drum_mapping_available(print_missing=True)])
        if not calculable:
            print("can't get hits as there is no hvo score previously provided")
        else:
            return self.hvo[:, :self.number_of_voices]

    @property
    def velocities(self):
        calculable = all([self.is_hvo_score_available(print_missing=True),
                          self.is_drum_mapping_available(print_missing=True)])
        if not calculable:
            print("can't get velocities as there is no hvo score previously provided")
        else:
            return self.hvo[:, self.number_of_voices: 2 * self.number_of_voices]

    @property
    def offsets(self):
        calculable = all([self.is_hvo_score_available(print_missing=True),
                          self.is_drum_mapping_available(print_missing=True)])
        if not calculable:
            print("can't get offsets/utimings as there is no hvo score previously provided")
        else:
            return self.hvo[:, 2 * self.number_of_voices: ]

    #   ----------------------------------------------------------------------
    #   Utility methods for segment derivation
    #   EACH SEGMENT MEANS A PART THAT TEMPO AND TIME SIGNATURE IS CONSTANT
    #   ----------------------------------------------------------------------

    @property
    def tempo_consistent_segment_boundaries(self):
        #   Returns time boundaries within which the tempo is constant
        #   upper bound (infinite) is always at 100000 seconds
        #   Example: tempo change at 1.5 seconds
        #            method returns --> [0, 1.5, 100000] --> meaning that tempo is
        #                               constant between 0≤t<1.5, 1.5≤t,100000
        #   If no tempo changes in the track (i.e. consistent tempo across all times
        #            method returns --> [0, 1000000]
        calculable = self.is_tempos_available(print_missing=True)
        if not calculable:
            warnings.warn("Can't carry out request as Tempos are not specified")
            return None
        else:
            time_regions = [0, 100000]      # 100000 to denote infinite
            for ix, tempo in enumerate(self.tempos):
                if ix > 0:  # Force 1st tempo to be at 0 even if doesn't start at the very beginning
                    time_regions.append(tempo.time_step)
            return list(np.unique(time_regions))

    @property
    def time_signature_consistent_segment_boundaries(self):
        #   Returns time boundaries within which the time signature is constant
        #   upper bound (infinite) is always at 100000 seconds
        #   Example: time signature change at 1.5 seconds
        #            method returns --> [0, 1.5, 100000] --> meaning that tempo is
        #                               constant between 0≤t<1.5, 1.5≤t,100000
        #   If no time signature changes in the track (i.e. consistent time signature across all times
        #            method returns --> [0, 1000000]
        calculable = self.is_time_signatures_available(print_missing=True)
        if not calculable:
            warnings.warn("Can't carry out request as Time Signatures are not specified")
            return None
        else:
            time_regions = [0, 100000]      # 100000 to denote infinite
            for ix, time_signature in enumerate(self.time_signatures):
                if ix > 0:  # Force 1st tempo to be at 0 even if doesn't start at the very beginning
                    time_regions.append(time_signature.time_step)
            return list(np.unique(time_regions))

    @property
    def segment_boundaries(self):
        #   Returns time boundaries within which the time signature and tempo are constant
        #   upper bound (infinite) is always at 100000 seconds
        #   Example: time signature change at 1.5 seconds and tempo change at 2 seconds
        #            method returns --> [0, 1.5, 2, 100000] --> meaning that tempo is
        #                               constant between 0≤t<1.5, 1.5≤t,100000
        #   If no time signature or tempo changes in the track:
        #            method returns --> [0, 1000000]
        calculable = all([self.is_time_signatures_available(print_missing=True),
                          self.is_tempos_available(print_missing=True)])
        if not calculable:
            warnings.warn("Can't carry out request as above fields are missing")
            return None
        else:
            _time_regions = list()
            _time_regions.extend(self.tempo_consistent_segment_boundaries)
            _time_regions.extend(self.time_signature_consistent_segment_boundaries)
            return list(np.unique(_time_regions))

    @property
    def tempo_consistent_segment_lower_bounds(self):
        boundaries = self.tempo_consistent_segment_boundaries
        if boundaries is not None:
            return boundaries[:-1]
        else:
            return None

    @property
    def time_signature_consistent_segment_lower_bounds(self):
        boundaries = self.time_signature_consistent_segment_boundaries
        if boundaries is not None:
            return boundaries[:-1]
        else:
            return None

    @property
    def segment_lower_bounds(self):
        boundaries = self.segment_boundaries
        if boundaries is not None:
            return boundaries[:-1]
        else:
            return None

    @property
    def segment_upper_bounds(self):
        # Returns exclusive upper bounds of each segment
        boundaries = self.segment_boundaries

        if boundaries is not None and self.hvo is not None:
            upper_bounds = boundaries[1:]
            upper_bounds[-1] = len(self.hvo)
            return upper_bounds
        else:
            return None

    # IN THE DOCUMENTATION MENTION THAT EACH SEGMENT MEANS A PART THAT TEMPO AND TIME SIGNATURE IS CONSTANT
    @property
    def tempos_and_time_signatures_per_segments(self):
        # Returns two lists: 1. lists of tempos per segment
        #                     2. lists of time signature for each segment
        # Segments are defined as parts of the score where the tempo and time signature don't change
        calculable = all([self.is_time_signatures_available(print_missing=True),
                          self.is_tempos_available(print_missing=True)])
        if not calculable:
            warnings.warn("Can't carry out request as above fields are missing")
            return None, None
        else:
            tempos = list()
            time_signatures = list()

            """segment_bounds = self.segment_boundaries
            lower_bounds = segment_bounds[:-1]"""

            for segment_lower_bound in self.segment_lower_bounds:
                # Recursively find what the tempo and time signature is at the lower bound
                distance_from_tempo_boundaries = np.array(self.tempo_consistent_segment_boundaries) - \
                                                 segment_lower_bound
                tempo_index = np.where(distance_from_tempo_boundaries <= 0,
                                       distance_from_tempo_boundaries, -np.inf).argmax()
                distance_from_time_sig_boundaries = np.array(self.time_signature_consistent_segment_boundaries) - \
                                                    segment_lower_bound

                time_signature_index = np.where(distance_from_time_sig_boundaries <= 0,
                                                distance_from_time_sig_boundaries, -np.inf).argmax()

                tempos.append(self.tempos[tempo_index])
                time_signatures.append(self.time_signatures[time_signature_index])

            return tempos, time_signatures

    @property
    def number_of_segments(self):
        # Returns the number of segments in each of which the tempo and time signature is consistent
        calculable = all([self.is_time_signatures_available(print_missing=True),
                          self.is_tempos_available(print_missing=True)])
        if not calculable:
            warnings.warn("Can't carry out request as above fields are missing")
            return None
        else:
            segment_bounds = self.segment_boundaries
            return len(segment_bounds) - 1

    @property
    def beat_durations_per_segments(self):

        # Calculates the duration of each beat in seconds if time signature and qpm are available
        calculable = all([self.is_tempos_available(print_missing=True),
                          self.is_time_signatures_available(print_missing=True)])
        if not calculable:
            warnings.warn("Beat durations per segments can be calculated as the above fields are missing")
            return None
        else:
            beat_durs_per_segment = list()
            segment_lower_boundaries = np.array(self.segment_lower_bounds)
            for ix, segment_lower_bound in enumerate(segment_lower_boundaries):
                tempo, time_signature = self.tempo_and_time_signature_at_step(segment_lower_bound)
                beat_durs_per_segment.append((60.0 / tempo.qpm) * 4.0 / time_signature.denominator)
            return beat_durs_per_segment

    @property
    def steps_per_beat_per_segments(self):
        # Calculates the total number of steps in each beat belonging
        # to each tempo & time signature consistent segment
        tempos_per_seg, time_signatures_per_seg = self.tempos_and_time_signatures_per_segments

        if tempos_per_seg is not None and time_signatures_per_seg is not None:
            steps_per_beat_per_segment = []
            for ix, time_signature_in_segment_ix in enumerate(time_signatures_per_seg):
                # calculate the number of steps per beat for corresponding beat_division_factors
                beat_divs = time_signature_in_segment_ix.beat_division_factors
                mock_beat_grid_lines = np.concatenate([np.arange(beat_div)/beat_div for beat_div in beat_divs])
                steps_per_beat_per_segment.append(len(np.unique(mock_beat_grid_lines)))

            return steps_per_beat_per_segment
        else:
            warnings.warn("Tempo or Time Signature missing")
            return None

    @property
    def steps_per_segments(self):
        # number of steps in each segment (tempo & time signature consistent segment)

        segment_lower_bounds = self.segment_lower_bounds
        segment_upper_bounds = self.segment_upper_bounds

        if segment_lower_bounds is not None and self.is_hvo_score_available(print_missing=True) is not None:
            return list(np.array(segment_upper_bounds) - np.array(segment_lower_bounds))
        else:
            warnings.warn("Tempo or Time Signature missing")
            return None

    @property
    def n_beats_per_segments(self):
        # Calculate the number of beats in each tempo and time signature consistent segment of score/sequence

        calculable = all([self.is_hvo_score_available(print_missing=True),
                          self.is_tempos_available(print_missing=True),
                          self.is_time_signatures_available(print_missing=True)])

        if calculable:
            n_beats_per_seg = list()

            steps_per_beat_per_segments = self.steps_per_beat_per_segments
            segment_boundaries = self.segment_boundaries
            # replace 100000 (upper bound for last segment) in segment_boundaries with actual length of hvo score
            segment_boundaries[-1] = self.total_number_of_steps

            for ix, steps_per_beat_per_segment_ix in enumerate(steps_per_beat_per_segments):
                total_steps_in_segment_ix = segment_boundaries[ix+1]-segment_boundaries[ix]
                beats_in_segment_ix = total_steps_in_segment_ix/steps_per_beat_per_segment_ix
                n_beats_per_seg.append(beats_in_segment_ix)

            return n_beats_per_seg

        else:
            return None

    @property
    def n_bars_per_segments(self):
        # Returns the number of bars in each of the tempo and time signature consistent segments
        n_beats_per_segments = self.n_beats_per_segments
        _, time_signatures = self.tempos_and_time_signatures_per_segments

        if n_beats_per_segments is not None and time_signatures is not None:
            n_bars_per_segments = list()
            for segment_ix, n_beats_per_segment_ix in enumerate(n_beats_per_segments):
                n_bars_in_segment_ix = n_beats_per_segment_ix/time_signatures[segment_ix].numerator
                n_bars_per_segments.append(n_bars_in_segment_ix)
            return n_bars_per_segments
        else:
            warnings.warn("Can't execute request as above fields are missing")
            return None

    @property
    def segment_durations(self):
        beat_durations_per_segments = self.beat_durations_per_segments

        if beat_durations_per_segments is not None:

            n_beats_per_segments = self.n_beats_per_segments

            segment_durations = list()

            for segment_ix, beat_durations_in_segment_ix in enumerate(beat_durations_per_segments):
                segment_durations.append(beat_durations_in_segment_ix * self.n_beats_per_segments[segment_ix])

            return segment_durations
        else:
            return None

    def tempo_segment_index_at_step(self, step_ix):
        # gets the index of the tempo segment in which the step is located
        if self.is_tempos_available(print_missing=True) is None:
            warnings.warn("Can't carry out request as above fields are missing")
            return None
        else:
            # Get the boundaries where tempo or time signature changes
            tempo_boundaries = np.array(self.tempo_consistent_segment_boundaries)
            # find the distance between time stamp and the boundaries of the tempo segments
            tempo_boundaries_distance = np.array(tempo_boundaries) - step_ix
            # Figure out to which tempo segment the step belongs
            #  corresponding region will be identified with the index of the last negative value
            tempo_ix = np.where(tempo_boundaries_distance <= 0, tempo_boundaries_distance, -np.inf).argmax()
            return tempo_ix

    def time_signature_segment_index_at_step(self, step_ix):
        # gets the index of the time_signature segment in which the step is located
        if self.is_time_signatures_available(print_missing=True) is None:
            warnings.warn("Can't carry out request as above fields are missing")
            return None
        else:
            # Get the boundaries where tempo or time signature changes
            time_signature_boundaries = np.array(self.time_signature_consistent_segment_boundaries)
            # find the distance between time stamp and the boundaries of the tempo segments
            time_signature_boundaries_distance = np.array(time_signature_boundaries) - step_ix
            # Figure out to which tempo segment the step belongs
            #  corresponding region will be identified with the index of the last negative value
            time_signature_ix = np.where(time_signature_boundaries_distance <= 0,
                                         time_signature_boundaries_distance, -np.inf).argmax()
            return time_signature_ix

    def segment_index_at_step(self, step_ix):
        # gets the index of the tempo and time_signature consistent segments in which the step is located
        calculable = all([self.is_time_signatures_available(print_missing=True),
                          self.is_tempos_available(print_missing=True)])
        if calculable is None:
            warnings.warn("Can't carry out request as above fields are missing")
            return None
        else:
            """# Get the boundaries where tempo or time signature changes
            segment_boundaries = np.array(self.segment_boundaries)
            # find the distance between time stamp and the boundaries of the tempo segments
            segment_boundaries_distance = np.array(segment_boundaries) - step_ix
            # Figure out to which tempo segment the step belongs
            #  corresponding region will be identified with the index of the last negative value
            segment_ix = np.where(segment_boundaries_distance <= 0,
                                                   segment_boundaries_distance, -np.inf).argmax()"""

            # find the correct segment --> correct segment is where the step is larger/eq to the lower bound
            #       and smaller than upper bound
            lower_boundaries = np.array(self.segment_lower_bounds)
            upper_boundaries = np.array(self.segment_upper_bounds)
            check_sides = np.where(lower_boundaries <= step_ix, True, False
                                   ) * np.where(upper_boundaries > step_ix, True, False)
            segment_ix = np.argwhere(check_sides == True)[0, 0]

            return segment_ix

    def tempo_and_time_signature_at_step(self, step_ix):
        # Figures out which tempo and time signature consistent segment the step belongs to
        # and then returns the corresponding tempo and time signature
        distance_from_tempo_boundaries = np.array(self.tempo_consistent_segment_boundaries) - step_ix
        tempo_index = np.where(distance_from_tempo_boundaries <= 0,
                               distance_from_tempo_boundaries, -np.inf).argmax()
        distance_from_time_sig_boundaries = np.array(self.time_signature_consistent_segment_boundaries) - step_ix
        time_signature_index = np.where(distance_from_time_sig_boundaries <= 0,
                                        distance_from_time_sig_boundaries, -np.inf).argmax()
        return self.tempos[tempo_index], self.time_signatures[time_signature_index]

    def step_position_from_segment_beginning(self, step_ix):
        # Returns the position of an step with respect to the beginning of the corresponding
        # tempo and time signature consistent segment

        # Find corresponding segment index for step_ix
        segment_ix = self.segment_index_at_step(step_ix)

        if segment_ix is None:
            warnings.warn("Can't carry out request as above fields are missing")
            return None
        else:
            # Return the distance of the index from the lower bound of segment
            return step_ix - self.segment_lower_bounds[segment_ix]

    def step_position_from_time_signature_segment_beginning(self, step_ix):
        # Returns the position of an step with respect to the beginning of the corresponding
        # segment in which the time_signature is constant

        # Find corresponding segment index for step_ix
        time_signature_segment_ix = self.time_signature_segment_index_at_step(step_ix)

        if time_signature_segment_ix is None:
            warnings.warn("Can't carry out request as above fields are missing")
            return None
        else:
            # Return the distance of the index from the lower bound of segment
            return step_ix - self.time_signature_consistent_segment_lower_bounds[time_signature_segment_ix]


    """@property
    def bar_lens_per_segments(self):
        # Returns """

    """@property
    def bar_len(self):
        # Calculates the duration of each bar in seconds if time signature and qpm are available
        calculable = all([self.is_tempos_available(print_missing=True),
                          self.is_time_signatures_available(print_missing=True)])
        if calculable:
            return self.beat_dur * self.time_signature["numerator"]
        else:
            return None"""

    @property
    def total_len(self):
        # Calculates the total length of score in seconds if hvo score, time signature and qpm are available
        calculable = all([self.is_hvo_score_available(print_missing=True),
                         self.is_tempos_available(print_missing=True),
                         self.is_time_signatures_available(print_missing=True)])
        if calculable:
            return self.grid_lines[-1]+0.5*(self.grid_lines[-1] - self.grid_lines[-2])
        else:
            return None

    @property
    def total_number_of_steps(self):
        # Calculates the total number of steps in the score/sequence
        calculable = self.is_hvo_score_available(print_missing=True)
        if calculable:
            return self.hvo.shape[0]
        else:
            return 0

    @property
    def grid_type_per_segments(self):
        # Returns a list of the type of grid per tempo and time signature consistent segment
        # Type at each segment can be:
        #   1. binary:  if the grid lines lie on 2^n divisions of each beat
        #   2. triplet: if the grid lines lie on divisions of each beat that are multiples of 3
        #   3. mix:     if the grid is a combination of binary and triplet

        calculable = all([self.is_tempos_available(print_missing=True),
                          self.is_time_signatures_available(print_missing=True)])

        if calculable:
            grid_types_per_segments = list()
            _, time_signatures_per_seg = self.tempos_and_time_signatures_per_segments
            for ix, time_sig_in_seg_ix in enumerate(time_signatures_per_seg):
                po2_grid_flag, triplet_grid_flag = None, None
                for factor in time_sig_in_seg_ix.beat_division_factors:
                    if is_power_of_two(factor):
                        po2_grid_flag = True
                    if factor % 3 == 0:
                        triplet_grid_flag = True

                if po2_grid_flag and triplet_grid_flag:
                    grid_types_per_segments.append( "mix")
                elif po2_grid_flag:
                    grid_types_per_segments.append("binary")
                else:
                    grid_types_per_segments.append("triplet")
            return grid_types_per_segments

        else:
            return None

    @property
    def is_grid_equally_distanced_per_segments(self):
        # for each tempo and time signature consistent segment
        #  Checks if the grid is uniformly placed (in case of binary or triplet grids)
        #       or if the grid is non-uniformly oriented (in case of mix grid)
        grid_type_per_segments = self.grid_type_per_segments
        if grid_type_per_segments is not None:
            is_grid_equally_distanced_per_segments = list()
            for grid_type_per_segment in grid_type_per_segments:
                is_grid_equally_distanced_per_segments.append(True if grid_type_per_segments == "mix" else False)
            return is_grid_equally_distanced_per_segments
        else:
            return None

    @property
    def major_and_minor_grid_line_indices(self):
        # Returns major and minor grid line indices (corresponding to 1st dimension of self.hvo and self.grid_lines)
        # Major lines lie on the beginning of each beat --> multiples of number of steps in each beat
        # Minor lines lie in between major gridlines    --> not multiples of number of steps in each beat

        calculable = all([self.is_hvo_score_available(print_missing=True),
                          self.is_tempos_available(print_missing=True),
                          self.is_time_signatures_available(print_missing=True)])

        if calculable:
            major_grid_line_indices = list()
            minor_grid_line_indices = list()

            # if a grid line index is divisible by the number of steps per beat, it is major (i.e. beat position)
            for grid_line_index in range(self.total_number_of_steps):

                segment_ix = self.segment_index_at_step(grid_line_index)
                position_in_segment = self.step_position_from_time_signature_segment_beginning(grid_line_index)

                if position_in_segment % self.steps_per_beat_per_segments[segment_ix] == 0:
                    major_grid_line_indices.append(grid_line_index)
                else:
                    minor_grid_line_indices.append(grid_line_index)

            return major_grid_line_indices, minor_grid_line_indices
        else:
            warnings.warn("Above fields are required for calculating major/minor grid line positions")
            return None, None

    @property
    def major_and_minor_grid_lines(self):
        # Returns major and minor grid lines
        # Major lines lie on the beginning of each beat --> multiples of number of steps in each beat
        # Minor lines lie in between major gridlines    --> not multiples of number of steps in each beat

        major_grid_line_indices, minor_grid_line_indices = self.major_and_minor_grid_line_indices

        if major_grid_line_indices is not None and minor_grid_line_indices is not None:

            # Get grid time stamps for the grid indices
            major_grid_lines = self.grid_lines[major_grid_line_indices]
            minor_grid_lines = self.grid_lines[minor_grid_line_indices]

            return major_grid_lines, minor_grid_lines
        else:
            warnings.warn("Above fields are required for calculating major/minor grid line positions")
            return None, None

    @property
    def downbeat_indices(self):
        # Returns the indices of the grid_lines where a downbeat occurs.
        major_grid_line_indices, minor_grid_line_indices = self.major_and_minor_grid_line_indices

        if major_grid_line_indices is not None and minor_grid_line_indices is not None:
            # Get grid time stamps for the grid indices
            steps_per_beat_per_segments = self.steps_per_beat_per_segments
            _, time_signatures = self.tempos_and_time_signatures_per_segments

            downbeat_indices = list()

            for major_grid_line_ix in major_grid_line_indices:
                segment_ix = self.segment_index_at_step(major_grid_line_ix)
                downbeat_indices.append(int(segment_ix/time_signatures[segment_ix].numerator))

            return downbeat_indices

        else:
            warnings.warn("Above fields are required for calculating downbeat positions (i.e. measure boundaries")
            return None

    @property
    def downbeat_positions(self):
        # Returns the time stamps for the downbeat positions in the grid
        downbeat_indices = self.downbeat_indices

        if downbeat_indices is not None:
            # Get grid time stamps for the downbeat indices
            downbeat_positions = self.grid_lines[downbeat_indices]
            return downbeat_positions
        else:
            warnings.warn("Above fields are required for calculating downbeat positions (i.e. measure boundaries")
            return None

    @property
    def starting_measure_indices(self):
        # A wrapper for downbeat_positions (for the sake of code readability)
        return self.downbeat_indices

    @property
    def starting_measure_positions(self):
        # A wrapper for downbeat_positions (for the sake of code readability)
        return self.downbeat_positions

    @property
    def grid_lines(self):
        # Creates the grid according to the features set for each tempo and time signature consistent segment

        # Get the number of steps in each tempo and time signature consistent segment
        n_steps_per_segments = self.steps_per_segments

        # Get tempos and time signatures for each segment
        tempos, time_signatures = self.tempos_and_time_signatures_per_segments

        # Get and round up the number of bars per segments
        n_bars_per_segments = np.ceil(self.n_bars_per_segments)

        # Keep track of the initial position of each segment grid section
        beginning_of_current_segment = 0

        # Variable for the final grid lines
        grid_lines = np.array([])

        for segment_ix, n_steps_per_segment in enumerate(n_steps_per_segments):
            # Create n-bars of grid lines
            segment_grid_lines, beginning_of_next_segment = create_grid_for_n_bars(
                n_bars=n_bars_per_segments[segment_ix],
                time_signature=time_signatures[segment_ix],
                tempo=tempos[segment_ix]
            )

            # Trim the grid to fit required number of steps, also shift lines to start at the beginning of segment
            trimmed_moved_segment = segment_grid_lines[:n_steps_per_segment] + beginning_of_current_segment

            # Set the beginning of next segment
            if len(segment_grid_lines) > n_steps_per_segment:
                # in case the time sig or tempo are changed before end of measure
                # This ideally shouldn't happen but here for the sake of completeness
                beginning_of_current_segment = segment_grid_lines[n_steps_per_segment]
            else:
                beginning_of_current_segment = beginning_of_next_segment

            grid_lines = np.append(grid_lines, trimmed_moved_segment)

        return grid_lines

    #   ----------------------------------------------------------------------
    #   Utility methods to check whether required properties are
    #       available for carrying out a request
    #   All methods have two args: print_missing, print_available
    #       set either to True to get additional info for debugging
    #
    #   Assuming that the local variables haven't been modified directly,
    #   No need to check the validity of data types if they are available
    #       as this is already done in the property.setters
    #   ----------------------------------------------------------------------

    def is_time_signatures_available(self, print_missing=False, print_available=False):
        # Checks whether time_signatures are already specified and necessary fields are filled
        time_signatures_ready_to_use = list()
        if self.time_signatures is not None:
            for time_signature in self.time_signatures:
                time_signatures_ready_to_use.append(time_signature.is_ready_to_use)

        if not all(time_signatures_ready_to_use):
            for ix, ready_status in enumerate(time_signatures_ready_to_use):
                if ready_status is not True:
                    print("There are missing fields in Time_Signature {}: {}".format(ix, self.time_signatures[ix]))
            return False
        else:
            return True

    def is_tempos_available(self, print_missing=False, print_available=False):
        # Checks whether tempos are already specified and necessary fields are filled
        tempos_ready_to_use = list()
        if self.tempos is not None:
            for tempo in self.tempos:
                tempos_ready_to_use.append(tempo.is_ready_to_use)

        if not all(tempos_ready_to_use):
            for ix, ready_status in enumerate(tempos_ready_to_use):
                if ready_status is not True:
                    print("There are missing fields in Tempo {}: {}".format(ix, self.tempos[ix]))
            return False
        else:
            return True

    def is_drum_mapping_available(self, print_missing=False, print_available=False):
        # Checks whether drum_mapping is already specified
        if not self.is_drum_mapping_available:
            if print_missing:
                print("\n|---- drum_mapping is not specified. Currently empty ")
            return False
        else:
            if print_available:
                print("\n|---- drum_mapping is available and specified as {}".format(self.drum_mapping))
            return True

    def is_hvo_score_available(self, print_missing=False, print_available=False):
        # Checks whether hvo score array is already specified
        if not self.is_drum_mapping_available:
            if print_missing:
                print("\n|---- HVO score is not specified. Currently empty ")
            return False
        else:
            if print_available:
                print("\n|---- HVO score is available and specified as {}".format(self.hvo))
            return True

    #   --------------------------------------------------------------
    #   Utilities to import/export different score formats such as
    #       1. NoteSequence, 2. HVO array, 3. Midi
    #   --------------------------------------------------------------

    def from_note_sequence(self, ns, beat_division_factors=None, max_n_bars=None):
        """
        # Note_Sequence importer. Converts the note sequence to hvo format
        @param ns:                  Note_Sequence drum score
        @param max_n_bars:          maximum number of bars to import
        @return:
        """
        convertible = list()
        convertible.append(self.is_time_signatures_available(print_missing=True))
        convertible.append(self.is_drum_mapping_available(print_missing=True))
        if not all(convertible):
            warnings.warn("Above fields are missing for initiating the HVO_Sequence from a NoteSequence score")
            warnings.warn("Update the above before making the request again")
            return None

        # Grab the note_sequence signatures
        signatures = list()

        for time_sig in ns.time_signatures:
            if time_sig.time:
                time_s = time_sig.time

            signatures.append(Time_Signature(time_sig.time, time_sig.numerator, time_sig.denominator,
                                             beat_division_factors=beat_division_factors))

        """signature_info = {
            "time": ns.time_signatures[0].time,
            "numerator": ns.time_signatures[0].numerator,           # AKA beats per bar
            "denominator": ns.time_signatures[0].denominator
        }
        """
        """
        # Grab the note_sequence tempos
        tempos = list()
        qpm = ns.tempos[0].qpm

        # Calculate the duration of each beat
        beat_dur = (60.0 / qpm) * signature_info["denominator"] / 4.0

        # Calculate the total length of sequence in seconds
        if max_n_bars is not None:
            max_len = max_n_bars * signature_info["numerator"] * beat_dur
            total_len = min(max_len, max(ns.notes[-1].end_time, ns.total_time))
        else:
            total_len = max(ns.notes[-1].end_time, ns.total_time)

        # Calculate the number of bars required to fit the sequence
        n_bars = np.round(total_len / beat_dur * signature_info["numerator"])

        # Create the grid time-stamps for hvo array
        grid_info = create_grid(self.beat_division_factors, n_bars, signature_info["numerator"], beat_dur)

        # Dimension at each time-step (3n) --> n hits, n velocities, and n utimings
        dimension_at_time_step = 3 * len(self.drum_mapping.keys())

        # Create empty hvo
        hvo = np.zeros((grid_info["loc"].shape[0], dimension_at_time_step))

        # Grab drum notes in note_sequence, and place each note in hvo array one-by-one
        for ix, note in enumerate(ns.notes):
            if note.is_drum:
                hvo = place_note_in_hvo(note, hvo, grid_info["loc"], self.drum_mapping)

        # Now that we've successfully reached here, we can safely update corresponding fields in the class
        self.time_signature = signature_info
        self.qpm = qpm
        self.hvo = hvo
        """
        return self

    def to_note_sequence(self, midi_track_n=9):
        """
        Exports the hvo_sequence to a note_sequence object

        @param midi_track_n:    the midi track channel used for the drum scores
        @return:
        """
        convertible = all([self.is_hvo_score_available(print_missing=True),
                           self.is_tempos_available(print_missing=True),
                           self.is_time_signatures_available(print_missing=True)])
        if not convertible:
            warnings.warn("Above fields need to be provided so as to convert the hvo_sequence into a note_sequence")
            return None

        # Create a note sequence instance
        ns = music_pb2.NoteSequence()

        # get the number of allowed drum voices
        n_voices = len(self.__drum_mapping.keys())

        # find nonzero hits tensor of [[position, drum_voice]]
        pos_instrument_tensors = np.transpose(np.nonzero(self.__hvo[:, :n_voices]))

        # Set note duration as 1/2 of the smallest grid distance
        note_duration = np.min(self.grid_lines[1:]-self.grid_lines[:-1]) / 2.0

        # Add notes to the NoteSequence object
        for drum_event in pos_instrument_tensors:  # drum_event -> [grid_position, drum_voice_class]
            grid_pos = drum_event[0]        # grid position
            drum_voice_class = drum_event[1]  # drum_voice_class in range(n_voices)

            # Grab the first note for each instrument group
            pitch = list(self.__drum_mapping.values())[drum_voice_class][0]
            velocity = self.__hvo[grid_pos, drum_voice_class + n_voices]  # Velocity of the drum event
            utiming_ratio = self.__hvo[                               # exact timing of the drum event (rel. to grid)
                grid_pos, drum_voice_class + 2 * n_voices]

            utiming = 0
            if utiming_ratio < 0:
                # if utiming comes left of grid, figure out the grid resolution left of the grid line
                if grid_pos > 0:
                    utiming = (self.grid_lines[grid_pos] - self.grid_lines[grid_pos-1]) * \
                              utiming_ratio
                else:
                    utiming = 0      # if utiming comes left of beginning,  snap it to the very first grid (loc[0]=0)
            elif utiming_ratio > 0:
                if grid_pos < (self.total_number_of_steps-2):
                    utiming = (self.grid_lines[grid_pos+1] -
                               self.grid_lines[grid_pos]) * utiming_ratio
                else:
                    utiming = (self.grid_lines[grid_pos] -
                               self.grid_lines[grid_pos-1]) * utiming_ratio
                    # if utiming_ratio comes right of the last grid line, use the previous grid resolution for finding
                    # the utiming value in ms

            start_time = self.grid_lines[grid_pos] + utiming      # starting time of note in sec

            end_time = start_time + note_duration                   # ending time of note in sec

            ns.notes.add(pitch=pitch, start_time=start_time.item(), end_time=end_time.item(),
                         is_drum=True, instrument=midi_track_n, velocity=int(velocity.item() * 127))

        ns.total_time = self.total_len

        for tempo in self.tempos:
            ns.tempos.add(
                time=self.grid_lines[tempo.time_step],
                qpm=tempo.qpm
            )

        for time_sig in self.time_signatures:
            ns.time_signatures.add(
                time=self.grid_lines[time_sig.time_step],
                numerator=time_sig.numerator,
                denominator=time_sig.denominator
            )

        return ns

    def save_hvo_to_midi(self, filename="misc/temp.mid", midi_track_n=9):
        """
            Exports to a  midi file

            @param filename:            filename/path for saving the midi
            @param midi_track_n:        midi track for

            @return pm:                 the pretty_midi object
        """
        convertible = all([self.is_hvo_score_available(print_missing=True),
                           self.is_tempos_available(print_missing=True),
                           self.is_time_signatures_available(print_missing=True)])
        if not convertible:
            warnings.warn("Above fields need to be provided so as to convert the hvo_sequence into a note_sequence")
            return None

        ns = self.to_note_sequence(midi_track_n=midi_track_n)
        pm = note_seq.note_sequence_to_pretty_midi(ns)
        pm.write(filename)
        return pm

    #   --------------------------------------------------------------
    #   Utilities to Synthesize the hvo score
    #   --------------------------------------------------------------

    def synthesize(self, sr=44100, sf_path="../hvo_sequence/soundfonts/Standard_Drum_Kit.sf2"):
        """
        Synthesizes the hvo_sequence to audio using a provided sound font
        @param sr:                          sample rate
        @param sf_path:                     path to the soundfont samples
        @return:                            synthesized audio sequence
        """
        synthesizable = all([self.is_hvo_score_available(print_missing=True),
                             self.is_tempos_available(print_missing=True),
                             self.is_time_signatures_available(print_missing=True)])
        if not synthesizable:
            warnings.warn("Above fields need to be provided so as to synthesize the sequence")
            return None

        ns = self.to_note_sequence(midi_track_n=9)
        pm = note_seq.note_sequence_to_pretty_midi(ns)
        audio = pm.fluidsynth(fs=sr, sf2_path=sf_path)
        return audio

    def save_audio(self, filename="misc/temp.wav", sr=44100,
                   sf_path="../hvo_sequence/soundfonts/Standard_Drum_Kit.sf2"):
        """
        Synthesizes and saves the hvo_sequence to audio using a provided sound font
        @param filename:                    filename/path used for saving the audio
        @param sr:                          sample rate
        @param sf_path:                     path to the soundfont samples
        @return:                            synthesized audio sequence
        """
        synthesizable = all([self.is_hvo_score_available(print_missing=True),
                             self.is_tempos_available(print_missing=True),
                             self.is_time_signatures_available(print_missing=True)])
        if not synthesizable:
            warnings.warn("Above fields need to be provided so as to synthesize the sequence")
            return None

        ns = self.to_note_sequence(midi_track_n=9)
        pm = note_seq.note_sequence_to_pretty_midi(ns)
        audio = pm.fluidsynth(sf2_path=sf_path, fs=sr)
        sf.write(filename, audio, sr, 'PCM_24')
        return audio

    #   --------------------------------------------------------------
    #   Utilities to plot the score
    #   --------------------------------------------------------------

    def to_html_plot(self, filename="misc/temp.html", show_figure=False):
        """
        Creates a bokeh plot of the hvo sequence
        @param filename:                    path to save the html plot
        @param show_figure:                 If True, opens the plot as soon as it's generated
        @return:                            html_figure object generated by bokeh
        """
        plottable = all([self.is_hvo_score_available(print_missing=True),
                         self.is_tempos_available(print_missing=True),
                         self.is_time_signatures_available(print_missing=True)])

        if not plottable:
            warnings.warn("Above fields need to be provided so as to synthesize the sequence")
            return None

        ns = self.to_note_sequence(midi_track_n=9)
        # Create the initial piano roll
        _html_fig = note_seq.plot_sequence(ns, show_figure=False)
        _html_fig.title.text = filename.split("/")[-1]  # add title

        # Add y-labels corresponding to instrument names rather than midi note ("kick", "snare", ...)
        unique_pitches = set([note.pitch for note in ns.notes])

        # Find corresponding drum tags
        drum_tags = []
        for p in unique_pitches:
            _, tag, _ = find_pitch_and_tag(p, self.__drum_mapping)
            drum_tags.append(tag)

        _html_fig.xgrid.grid_line_color = None
        _html_fig.ygrid.grid_line_color = None

        _html_fig.yaxis.ticker = list(unique_pitches)
        _html_fig.yaxis.major_label_overrides = dict(zip(unique_pitches, drum_tags))

        # Add beat and beat_division grid lines
        major_grid_lines, minor_grid_lines = self.major_and_minor_grid_lines

        for t in minor_grid_lines:
            minor_grid_ = Span(location=t, dimension='height', line_color='black', line_width=.1)
            _html_fig.add_layout(minor_grid_)

        for t in major_grid_lines:
            major_grid_ = Span(location=t, dimension='height', line_color='blue', line_width=.5)
            _html_fig.add_layout(major_grid_)

        for t in self.starting_measure_positions:
            major_grid_ = Span(location=t, dimension='height', line_color='blue', line_width=2)
            _html_fig.add_layout(major_grid_)

        # Plot the figure if requested
        if show_figure:
            show(_html_fig)

        # Save the plot
        output_file(filename)  # Set name used for saving the figure
        save(_html_fig)  # Save to file

        return _html_fig




