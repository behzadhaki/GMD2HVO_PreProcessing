import copy
import note_seq

def polysyncs_per_loops_using_hvo_sequence(hvo_sequence_list):
    sync_list = []
    for hvo_seq in hvo_sequence_list:
        ns = hvo_seq.to_note_sequence()
        temp_note_seq = quantize(ns)
        polysync, totalpolysinc = polysync_note_seq(temp_note_seq)

        # sync_list.append(float(syncopation32(summed))/float(len(temp_note_seq.notes))) #
        sync_list.append(totalpolysinc)

    return sync_list

def polysyncs_per_loops_using_note_sequence(note_sequence_list):
    sync_list = []
    for ns in note_sequence_list:
        temp_note_seq = quantize(ns)
        polysync, totalpolysinc = polysync_note_seq(temp_note_seq)

        # sync_list.append(float(syncopation32(summed))/float(len(temp_note_seq.notes))) #
        sync_list.append(totalpolysinc)

    return sync_list

def quantize(ns):
        temp_note_seq = copy.deepcopy(ns)
        temp_note_seq = note_seq.sequences_lib.quantize_note_sequence(temp_note_seq, 4)
        return temp_note_seq

def polysync_note_seq(quantised_note_seq):
    # Descriptor 16: polysync, polyphonic syncopation based on Witeks formula
    low_pitches = [35, 36, 41, 45, 47, 64, 66, 77]
    mid_pitches = [37, 38, 39, 40, 43, 50, 61, 65, 68, 78, 79]
    high_pitches = [42, 44, 46, 48, 49, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 62, 63, 67,
                    69, 70, 71, 72, 73, 74, 75, 76, 80, 81, 22, 26]

    lopatt = [0] * 32
    midpatt = [0] * 32
    hipatt = [0] * 32

    for note in quantised_note_seq.notes:
        # print(note.pitch, ' , ', note.quantized_start_step)
        if note.quantized_start_step > 31:
            if note.quantized_start_step > 32:
                print('More than 32 step wtf???, time signature: ', quantised_note_seq.time_signatures)
                for note in quantised_note_seq.notes:
                    print(note.pitch, ' , ', note.quantized_start_step)
                raise ValueError;
            continue;

        if note.pitch in low_pitches:
            lopatt[note.quantized_start_step] = 1
        elif note.pitch in mid_pitches:
            midpatt[note.quantized_start_step] = 1
        elif note.pitch in high_pitches:
            hipatt[note.quantized_start_step] = 1
        else:
            print('this pitch doesnt exist wtf: ', note.pitch)

    nonsilence = []
    salienceprofile = [5, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1]

    # compute POLYPHONIC SYNCOPATION as proposed by witek et al. 2014
    # iterate through all three patterns and consolidate a non silence pattern with (index, elements, salience)
    for s, step in enumerate(lopatt):
        pack = []
        if lopatt[s] != 0:
            pack.append('l')
        if midpatt[s] != 0:
            pack.append('m')
        if hipatt[s] != 0:
            pack.append('h')
        if len(pack) != 0:
            nonsilence.append((s, pack, salienceprofile[s]))

            # print pattname, pattern
    # print pattname, nonsilence
    polysync = []
    for s, step in enumerate(nonsilence):
        if step[1] != nonsilence[(s + 1) % len(nonsilence)][1]:  # los dos steps no pueden ser iguales
            # caso del kick vs snare
            if 'l' in step[1] and 'm' in nonsilence[(s + 1) % len(nonsilence)][1] and step[2] <= \
                    nonsilence[(s + 1) % len(nonsilence)][2]:
                # print 'yes', step[0], abs(step[2] - nonsilence[(s+1)%len(nonsilence)][2])+2
                syncpack = step[0], abs(step[2] - nonsilence[(s + 1) % len(nonsilence)][2]) + 2, 'c1'
                polysync.append(syncpack)

            # caso del kick vs hh
            if 'l' in step[1] and 'h' in nonsilence[(s + 1) % len(nonsilence)][1] and 'm' not in \
                    nonsilence[(s + 1) % len(nonsilence)][1] and step[2] <= nonsilence[(s + 1) % len(nonsilence)][2]:
                # print 'yes', step[0], abs(step[2] - nonsilence[(s+1)%len(nonsilence)][2])+2
                syncpack = step[0], abs(step[2] - nonsilence[(s + 1) % len(nonsilence)][2]) + 5, 'c2'
                polysync.append(syncpack)

            # caso del sn vs kick
            if 'm' in step[1] and 'l' in nonsilence[(s + 1) % len(nonsilence)][1] and step[2] <= \
                    nonsilence[(s + 1) % len(nonsilence)][2]:
                # print 'yes', step[0], abs(step[2] - nonsilence[(s+1)%len(nonsilence)][2])+2
                syncpack = step[0], abs(step[2] - nonsilence[(s + 1) % len(nonsilence)][2]) + 1, 'c3'
                polysync.append(syncpack)

            # caso del sn vs hh
            if 'm' in step[1] and 'h' in nonsilence[(s + 1) % len(nonsilence)][1] and 'l' not in \
                    nonsilence[(s + 1) % len(nonsilence)][1] and step[2] <= nonsilence[(s + 1) % len(nonsilence)][2]:
                # print 'yes', step[0], abs(step[2] - nonsilence[(s+1)%len(nonsilence)][2])+2
                syncpack = step[0], abs(step[2] - nonsilence[(s + 1) % len(nonsilence)][2]) + 5, 'c4'
                polysync.append(syncpack)

    # print pattname, nonsilence
    # print pattname, polysync
    totalpolysinc = 0
    for s in polysync:
        totalpolysinc = totalpolysinc + s[1]
    return polysync, totalpolysinc