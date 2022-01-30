import gzip
import os
import pickle
import warnings

import numpy as np
import wfdb

# https://wfdb.readthedocs.io/en/latest/io.html#wfdb.io.show_ann_labels
label_mapping = {"btype": {0: ('Q', ''),       # Undefined: Unclassifiable beat
                           1: ('N', ''),       # Normal: Normal beat
                           2: ('S', ''),       # ESSV (PAC): Premature or ectopic supraventricular beat
                           3: ('a', ''),       # Aberrated: Aberrated atrial premature beat
                           4: ('V', '')},      # ESV (PVC): Premature ventricular contraction
                 "rtype": {0: ('', ''),        # Null/Undefined
                           1: ('', ''),        # End
                           2: ('', ''),        # Noise
                           3: ('+', "(N"),     # NSR (normal sinusal rhythm): Normal sinusal rhythm
                           4: ('+', "(AFIB"),  # AFib: Atrial fibrillation
                           5: ('+', "(AFL"),   # AFlutter: Atrial flutter
                           6: (None, None)}}   # Used to split a rhythm when a beat annotation is not
                                               # linked to a rhythm type


def get_person_attributes(path):
    dirname, filename = os.path.split(path)
    filename, ext = filename.split('.')[0], '.'.join(filename.split('.')[1:])
    idx = int(filename.split('_')[0])
    # XXXXX_batched.pkl.gz
    data_path = os.path.join(dirname, f"{filename}.{ext}")
    # XXXXX_batched_lbls.pkl.gz
    labels_path = os.path.join(dirname, f"{filename}_lbls.pkl.gz")
    return idx, data_path, labels_path


def make_wfdb(path):
    """Create a WFDB file from a .../XXXXX_batched.pkl.gz"""
    person_idx, person_data_path, person_labels_path = \
        get_person_attributes(path)

    assert os.path.exists(person_data_path)
    assert os.path.exists(person_labels_path)

    print(f"Make `p{person_idx:05d}/p{person_idx:05d}_s*` from "
          f"`{person_data_path}` and `{person_labels_path}`")

    with gzip.open(person_data_path) as gzf:
        person_data = pickle.load(gzf)
    with gzip.open(person_labels_path) as gzf:
        person_labels = pickle.load(gzf)

    cwd = os.getcwd()

    try:
        # wfdb.io.wrann does not allow '/' in path which forces the cwd to be
        # changed to the person's subdir
        os.makedirs(f"p{person_idx:05d}", exist_ok=True)
        os.chdir(f"p{person_idx:05d}")
        for i, (p_signal, seg_labels) in enumerate(zip(person_data,
                                                       person_labels)):
            # https://wfdb.readthedocs.io/en/latest/io.html#wfdb.io.wrsamp
            wfdb.io.wrsamp(f"p{person_idx:05d}_s{i:02d}", fs=250, units=["mV"],
                           # Use channel index as signal name
                           sig_name=["ecg"],
                           p_signal=p_signal.reshape(-1, 1),
                           # Use int16 to store the digital signal as it is the
                           # original data format
                           fmt=["16"],
                           # Scale the physical signal to fill most of the range
                           # provided by the data format (int16)
                           # p_signal: [-440.625, 440.625]
                           # adc_gain = max_positive * ads_gain / (calibration_factor * rail_to_rail_voltage) ~= 74.36
                           adc_gain=[(2**(16 - 1) - 1) * 6 / (0.9375 * 2820)],
                           baseline=[0], write_dir=".")

            beats = np.unique(np.concatenate(seg_labels["btype"]))
            rhythms = np.unique(np.concatenate(seg_labels["rtype"]))
            if not np.in1d(beats, rhythms).all():
                warnings.warn("Beat annotations count does not match with "
                    "rhythm annotations count for p{:05d}_s{:02d}. Spliting "
                    "rhythm zones surrounding extra beats".format(
                        person_idx, i))
                seg_labels["rtype"].append(beats[np.in1d(beats, rhythms) == False])

            flat_ann = []
            for label_type, labels in seg_labels.items():
                for type_id, locs in enumerate(labels):
                    if locs.size and label_type == "rtype" and type_id == 0:
                        warnings.warn("Undefined rhythm annotation should not "
                            "exist in icentia11k for p{:05d}_s{:02d}".format(
                                person_idx, i))
                    flat_ann.extend([(loc, label_type,
                                      *label_mapping[label_type][type_id])
                                     for loc in locs])
            if not flat_ann:
                warnings.warn("Empty annotations for p{:05d}_s{:02d}".format(
                    person_idx, i))
                continue
            flat_ann.sort()

            sample, ann_type, symbol, _aux_note = zip(*flat_ann)
            sample = np.asarray(sample)
            symbol = np.asarray(symbol)
            chan = np.asarray([0] * sample.size)
            aux_note = np.asarray(_aux_note, dtype="<U6")

            rtype = np.where(np.asarray(ann_type) == "rtype")[0]
            for _i in range(1, len(rtype) + 1):
                prev_idx, idx = (rtype[_i-1],
                                 rtype[_i] if _i < len(rtype) else None)
                prev_aux, aux = (_aux_note[prev_idx],
                                 _aux_note[idx] if idx is not None else None)
                if aux and prev_aux == aux:
                    symbol[idx] = ''
                    aux_note[idx] = ''
                elif prev_aux:
                    # Single beat in rhythm zone
                    if symbol[prev_idx] == '+':
                        aux_note[prev_idx] += ')'
                    # Closing rhythm zone
                    else:
                        symbol[prev_idx] = '+'
                        aux_note[prev_idx] = ')'

            mask = (symbol != '') * (symbol != None)
            sample = sample[mask]
            symbol = symbol[mask]
            aux_note = aux_note[mask]
            chan = np.array([0] * sample.size)
            # https://wfdb.readthedocs.io/en/latest/io.html#wfdb.io.wrann
            wfdb.io.wrann(f"p{person_idx:05d}_s{i:02d}", "atr", sample,
                          symbol=symbol, chan=chan, aux_note=aux_note, fs=250,
                          custom_labels=None, write_dir=".")
    finally:
        # chdir back to initial cwd
        os.chdir(cwd)

    validate_wfdb(path)


def validate_wfdb(path):
    person_idx, person_data_path, person_labels_path = \
        get_person_attributes(path)
    wfdb_person_dir = f"p{person_idx:05d}"

    print(f"Validate `p{person_idx:05d}/p{person_idx:05d}_s*` with "
          f"`{person_data_path}` and `{person_labels_path}`")

    assert os.path.exists(person_data_path)
    assert os.path.exists(person_labels_path)

    with gzip.open(person_data_path) as gzf:
        person_data = pickle.load(gzf)
    with gzip.open(person_labels_path) as gzf:
        person_labels = pickle.load(gzf)

    for i, (p_signal, seg_labels) in enumerate(zip(person_data,
                                                   person_labels)):
        record_path = os.path.join(wfdb_person_dir, f"p{person_idx:05d}_s{i:02d}")
        assert p_signal.size == wfdb.rdrecord(record_path).p_signal.size

        labels = {"btype": {'Q': np.unique(seg_labels["btype"][0]).size,
                            'N': np.unique(seg_labels["btype"][1]).size,
                            'S': np.unique(seg_labels["btype"][2]).size,
                            'a': np.unique(seg_labels["btype"][3]).size,
                            'V': np.unique(seg_labels["btype"][4]).size},
                  "rtype": {'': sum([np.unique(l).size for l in
                                     seg_labels["rtype"][0:3]]),
                            "(N": np.unique(seg_labels["rtype"][3]).size,
                            "(AFIB": np.unique(seg_labels["rtype"][4]).size,
                            "(AFL": np.unique(seg_labels["rtype"][5]).size}}
        unique_b_o = np.unique(np.concatenate([l for l in seg_labels["btype"]]))
        unique_r_o = np.unique(np.concatenate([l for l in seg_labels["rtype"]]))

        ann_path = os.path.join(wfdb_person_dir, f"p{person_idx:05d}_s{i:02d}")
        try:
            ann = wfdb.io.rdann(ann_path, "atr")
            sample = ann.sample
            symbol = np.asarray(ann.symbol)
            aux = [_ for _ in zip(sample[symbol == '+'],
                                  np.asarray(ann.aux_note)[symbol == '+'])]
        except FileNotFoundError:
            ann = None
            sample = np.empty(0)
            symbol = np.empty(0, dtype="<U6")
            aux = []

        unique_samples = np.unique(sample[symbol != '+'])
        expand_r = np.array(sample, dtype="<U6")
        expand_r[:] = ''
        for i in range(0, len(aux)):
            if aux[i][1] == ')':
                low, high = aux[i-1][0], aux[i][0]
                expand_r[(ann.sample >= low) == (sample <= high)] = \
                    aux[i-1][1]
            # Single beat in rhythm zone
            elif aux[i][1].startswith('(') and aux[i][1].endswith(')'):
                expand_r[sample == aux[i][0]] = aux[i][1][:-1]

        counts = {_s: 0 for _s in labels["btype"].keys()}
        for k in counts:
            _samples = np.unique(sample[symbol == k])
            counts[k] += _samples.size
        assert counts == labels["btype"]

        counts = {_s: 0 for _s in labels["rtype"].keys()}
        for k in counts:
            _samples = np.unique(sample[(symbol != '+') * (expand_r == k)])
            counts[k] += _samples.size

        # Some samples are found multiple times in the beats and rhythms annotations
        if np.all(unique_samples == unique_r_o):
            assert {**counts, '': 0} == {**labels["rtype"], '': 0}
        # Some rhythms are not linked to a beat in original data
        else:
            extra_rhythms = unique_r_o[np.in1d(unique_r_o, unique_b_o) == False]
            _noise = np.unique(np.concatenate([l for l in seg_labels["rtype"][0:3]]))
            _n = np.unique(seg_labels["rtype"][3])
            _afib = np.unique(seg_labels["rtype"][4])
            _afl = np.unique(seg_labels["rtype"][5])
            for t, l in [("(N", _n), ("(AFIB", _afib), ("(AFL", _afl)]:
                assert counts[t] == l[np.in1d(l, extra_rhythms) == False].size
            assert (
                # Some beats are not included in a rhythm zone in original data
                counts[''] - unique_b_o[np.in1d(unique_b_o, unique_r_o) == False].size ==
                _noise[np.in1d(_noise, extra_rhythms) == False].size)


make_wfdb("icentia11k/06039_batched.pkl.gz")