import sys
import numpy as np
import pickle
import gzip
from itertools import chain
import random


def load_file(filename):
    # print(">", filename)
    data = pickle.load(gzip.open(filename))
    return data

def stream_array(data, chunk_size=5, shuffle=True):
    if shuffle:
        np.random.shuffle(data)
    samples = (data.shape[0] // chunk_size) * chunk_size
    data = data[:samples]
    data = data.reshape(-1, chunk_size, data.shape[1])
    for i in range(data.shape[0]):
        yield data[i]


def buffered_random(stream, buffer_items=100, leak_percent=0.9):
    item_buffer = [None] * buffer_items
    leak_count = int(buffer_items * leak_percent)
    item_count = 0
    for item in stream:
        item_buffer[item_count] = item
        item_count += 1
        if buffer_items == item_count:
            random.shuffle(item_buffer)
            for item in item_buffer[leak_count:]:
                yield item
            item_count = leak_count
    if item_count > 0:
        item_buffer = item_buffer[:item_count]
        random.shuffle(item_buffer)
        for item in item_buffer:
            yield item


def stream_file_list(filenames, buffer_count=20, batch_size=10,
                     chunk_size=1,
                     shuffle=True):
    filenames = filenames.copy()
    if shuffle:
        random.shuffle(filenames)
    result = []
    streams = []
    while len(streams) < buffer_count and len(filenames) > 0:
        try:
            streams.append(stream_array(load_file(filenames.pop()),
                                        shuffle=shuffle,
                                        chunk_size=chunk_size))
        except IOError:
            pass
        except EOFError:
            pass
    while len(streams) > 0 or len(filenames) > 0:
        i = 0
        while len(result) < batch_size and len(streams) > 0:
            try:
                i = i % len(streams)
                next_item = next(streams[i])
                i = (i + 1) % len(streams)
                result.append(next_item)
            except StopIteration:
                # print("Need to looad new file")
                stream = None
                while len(filenames) > 0:
                    try:
                        stream = stream_array(load_file(filenames.pop()),
                                              shuffle=shuffle,
                                              chunk_size=chunk_size)
                        break
                    except IOError:
                        pass
                    except EOFError:
                        pass
                if stream is not None:
                    streams[i] = stream
                else:
                    streams = streams[:i] + streams[i+1:]
        if len(result) > 0:
            if all(x.shape == result[0].shape for x in result):
                yield np.stack(result)
            result = []
        if shuffle:
            random.shuffle(streams)


if __name__ == "__main__":
    import glob
    from pprint import pprint
    directory = "/home/shawntan/projects/rpp-bengioy/jpcohen/icentia12k"
    print(directory + "/*_batched.pkl.gz")
    filenames = glob.glob(directory + "/*_batched.pkl.gz")
    print(filenames)
    stream = stream_file_list(filenames)
    print(sum(x.shape[0] for x in stream))
